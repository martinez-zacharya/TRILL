import os
import sys
import time
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.rms import RMSD, RMSF
import prolif as plf
from openmm import CustomExternalForce
from openmm.app import PME, NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, LJPME
from openmm.app import NoCutoff, Modeller, StateDataReporter, PDBReporter
from openmm.app import PDBFile, Topology
from openmm.app.forcefield import ForceField
from openmm.app.simulation import Simulation
from openmm.openmm import LangevinMiddleIntegrator, MonteCarloBarostat, MinimizationReporter
from openmm.unit import kelvin, picosecond, picoseconds, femtosecond
from openmm.unit import nanometer, atmospheres
from openff.toolkit.topology import Molecule
from openff.toolkit import Topology as fftop
from openmm.vec3 import Vec3
from openff.toolkit import  Molecule
from openmm import unit
# from openmmforcefields.generators import SMIRNOFFTemplateGenerator
# from openmmforcefields.generators import SystemGenerator
from openmmforcefields_template_generators import SMIRNOFFTemplateGenerator
from openff.interchange import Interchange
from openff.toolkit import ForceField as ff_ff
from openff.toolkit.utils import get_data_file_path
from openmm.unit.quantity import Quantity
from tqdm import tqdm
from icecream import ic
from loguru import logger
import numpy as np
import copy
from openff.interchange import Interchange
from openmm import app
from rdkit import Chem
import parmed as pmd
import subprocess
from openff.interchange.components._packmol import RHOMBIC_DODECAHEDRON, pack_box
from .cuda_utils import set_platform_properties
import os
import pandas as pd


def parse_generalized_born_section(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    start_idx, end_idx = None, None
    for i, line in enumerate(lines):
        if line.strip() == "GENERALIZED BORN:":
            start_idx = i + 4  # Skip the headers
        elif start_idx is not None and line.strip().startswith("TOTAL"):
            end_idx = i + 1  # Include TOTAL line
            break

    if start_idx is None or end_idx is None:
        raise ValueError("Could not locate GENERALIZED BORN section")

    # Extract the data lines
    data_lines = lines[start_idx:end_idx]
    data = []
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) >= 4 and parts[1].replace('.', '', 1).replace('-', '', 1).isdigit():
            name = parts[0]
            avg = float(parts[1])
            stddev = float(parts[2])
            stderr = float(parts[3])
            data.append((name, avg, stddev, stderr))

    df = pd.DataFrame(data, columns=["Energy Component", "Average", "Std. Dev.", "Std. Err. of Mean"])
    return df

def write_mmpbsa_input_file(args):
    content = """Input file for running PB and GB
&general
   verbose=1,
/
&gb
  igb=5,
/
"""
    filepath = os.path.join(args.outdir, f"{args.name}_mmpbsa_input.in")
    with open(filepath, "w") as f:
        f.write(content)

    return filepath

def run_mmpbsa(input_file, solvated_prmtop, complex_prmtop, trajectory, output_file):
    """
    Run MMPBSA.py with the specified input file, solvated and complex prmtop files, and trajectory file.
    """
    cmd = [
        "MMPBSA.py",
        "-i", input_file,
        "-sp", solvated_prmtop,
        "-cp", complex_prmtop,
        "-y", trajectory,
        "-O",
        "-o", output_file
    ]

    try:
        subprocess.run(cmd, check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        print("MMPBSA.py completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error running MMPBSA.py:", e)
        
def autoimage_center_image_trajout(prmtop_path, pdb_in, pdb_out):
    cpptraj_input = f"""trajin {pdb_in}
autoimage
center :1-99999 mass origin
image origin center
trajout {pdb_out} pdb
"""
    subprocess.run(
        ["cpptraj", prmtop_path],
        input=cpptraj_input.encode(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

def strip_waters_ions(prmtop_path, output_prmtop):
    cpptraj_input = f"""
strip :WAT,Na+,Cl-
parmwrite out {output_prmtop}
"""
    subprocess.run(
        ["cpptraj", prmtop_path],
        input=cpptraj_input.encode(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

def write_nobox_prmtop(prmtop_path, output_prmtop_nobox):
    cpptraj_input = f"""parm {prmtop_path}
parmwrite out {output_prmtop_nobox} nobox
"""
    subprocess.run(
        ["cpptraj", prmtop_path],
        input=cpptraj_input.encode(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


def compute_interaction_fingerprint_postsim(top_path, traj_path, ligand_indices, initial_atom_indices, step_size):
    step_size = step_size / 1000
    u = mda.Universe(top_path, traj_path, dt=step_size)
    max_valid_index = u.atoms.n_atoms - 1
    ligand_indices_filtered = [i for i in ligand_indices if i <= max_valid_index]
    ligand_selection = u.atoms[ligand_indices_filtered]
    protein_selection = u.atoms[initial_atom_indices]

    interactions = [
    'Anionic',
    'CationPi',
    'Cationic',
    'EdgeToFace',
    'FaceToFace',
    'HBAcceptor',
    'HBDonor',
    'Hydrophobic',
    'MetalAcceptor',
    'MetalDonor',
    'PiCation',
    'PiStacking',
    'VdWContact',
    'XBAcceptor',
    'XBDonor']
    fp_count = plf.Fingerprint(interactions, count=True)

    fp_count.run(u.trajectory, ligand_selection, protein_selection, progress=True)
    df = fp_count.to_dataframe()
    df.columns = [f"{residue}|{interaction}|{protein}" for residue, interaction, protein in df.columns]
    df = df.reset_index()
    df['Frame'] = df['Frame'] + 1

    return df


# import martini_openmm as martini

def compute_rms_metrics_protein_ligand_exclude_na_cl(topology_file, trajectory_file, step_size):
    step_size = step_size / 1000
    u = mda.Universe(topology_file, trajectory_file, dt=step_size)

    # Selection excluding only sodium (NA) and chloride (CL) ions
    sel_complex = u.select_atoms("backbone")
    
    # Align to protein alpha carbons (assumes protein is present)
    align.AlignTraj(u, u, select="all", in_memory=True).run()

    # RMSD of full system minus Na/Cl
    rmsd_analyzer = RMSD(u, select="not (resname NA or resname CL)")
    rmsd_analyzer.run()
    rmsd_complex = rmsd_analyzer.results.rmsd[:, 2]

    # RMSF of non-ion atoms
    rmsf_analyzer = RMSF(sel_complex).run()
    rmsf_complex = rmsf_analyzer.results.rmsf
    return {
        "rmsd_complex": rmsd_complex,
        "rmsf_complex": rmsf_complex
    }

class SilentOutputStream:
    def write(self, *args, **kwargs):
        pass  # Simply discard the output

    def flush(self):
        pass

class ProgressBarReporter(StateDataReporter):
    def __init__(self, reportInterval, totalSteps, out=None, **kwargs):
        if out is None:
            out = sys.stdout  # Default to sys.stdout if no output stream is provided
        super().__init__(out, reportInterval, progress=True, totalSteps=totalSteps, **kwargs)
        self.totalSteps = totalSteps
        self.lastPercentageReported = -1
        self.startTime = time.time()
        self.lastReportTime = self.startTime
        self.estimatedEndTime = None

    def update_progress_bar(self, currentStep):
        percentage = (currentStep / self.totalSteps) * 100
        currentTime = time.time()
        elapsedTime = currentTime - self.startTime

        if int(percentage) % 1 == 0 and int(percentage) != self.lastPercentageReported:
            # Calculate estimated time to completion
            timePerPercentage = elapsedTime / percentage
            remainingTime = timePerPercentage * (100 - percentage)
            filledLength = int(50 * percentage // 100)
            bar = '█' * filledLength + '-' * (50 - filledLength)
            sys.stdout.write(f'\rEstimated Progress: |{bar}| {percentage:.2f}% ~{remainingTime // 60:.0f}m')
            sys.stdout.flush()
            self.lastPercentageReported = int(percentage)
            self.lastReportTime = currentTime

    def report(self, simulation, state):
        super().report(simulation, state)
        self.update_progress_bar(simulation.currentStep)
        

def combine_pdb_files(receptor_file, ligand_file, output_file):
    receptor_pdb = PDBFile(receptor_file)
    ligand_pdb = PDBFile(ligand_file)

    combined_topology = Topology()
    combined_positions = []

    # Copy over atoms and positions from receptor
    for chain in receptor_pdb.topology.chains():
        new_chain = combined_topology.addChain("R")
        for residue in chain.residues():
            new_residue = combined_topology.addResidue(residue.name, new_chain)
            for atom in residue.atoms():
                new_atom = combined_topology.addAtom(atom.name, atom.element, new_residue)
    combined_positions.extend(receptor_pdb.positions)

    # Copy over atoms and positions from ligand
    for chain in ligand_pdb.topology.chains():
        new_chain = combined_topology.addChain("L")
        for residue in chain.residues():
            new_residue = combined_topology.addResidue(residue.name, new_chain)
            for atom in residue.atoms():
                new_atom = combined_topology.addAtom(atom.name, atom.element, new_residue)
    combined_positions.extend(ligand_pdb.positions)

    # Write the combined PDB file
    with open(output_file, 'w') as f:
        PDBFile.writeFile(combined_topology, combined_positions, f)

    return output_file

def relax_structure(args, fixed_pdb_files):
    platform, properties = set_platform_properties(args)
    ic(fixed_pdb_files)
    for filename in tqdm(fixed_pdb_files, desc="Relaxing PDBs"):
        simulation = set_simulation_parameters(filename, platform, properties, args)
        minimize_and_save(simulation, args.outdir, filename)

def set_simulation_parameters(pdb_path, platform, properties, args):
    pdb = PDBFile(pdb_path)
    forcefield = ForceField(args.forcefield, args.solvent)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff)
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)
    return simulation


def minimize_and_save(simulation, output_dir, name):
    simulation.minimizeEnergy(maxIterations=0)
    filename = "relaxed_" + os.path.splitext(os.path.basename(name))[0] + '.pdb'
    relaxed_path = os.path.join(output_dir, filename)
    with open(relaxed_path, 'w+') as f:
        PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)
    simulation.context.reinitialize(preserveState=True)
    return simulation

# def setup_openff_interchange(args):
#     # rec_path = get_data_file_path('/home/zachmartinez/trill1.7.0/TRILL/MBP2.pdb')
#     topology = fftop.from_pdb(args.protein)
#     protein = topology.molecule(0)
#     lig = Molecule.from_file(args.ligand)
#     lig.generate_conformers(n_conformers=1)

#     topology = fftop.from_molecules([lig])
#     amber_force_field = ff_ff("ff14sb_off_impropers_0.0.3.offxml")
#     rec_inter = amber_force_field.create_interchange(topology=protein.to_topology())
#     lig_ff = ff_ff('openff-2.1.0.offxml', 'ff14sb_off_impropers_0.0.3.offxml')

#     if args.solvate:
#         water = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")
#         topology = pack_box(molecules=[lig, water, protein],number_of_copies=[1, 1000, 1],box_vectors=7.5 * RHOMBIC_DODECAHEDRON * unit.nanometer)
#         interchange = Interchange.from_smirnoff(force_field=lig_ff, topology=topology)
#         openmm_system = interchange.to_openmm()
#         openmm_topology = interchange.to_openmm_topology()
#         openmm_positions = interchange.positions.to_openmm()
#     else:

#         lig_inter = Interchange.from_smirnoff(lig_ff, topology)

#         interchange = Interchange.combine(rec_inter, lig_inter)
#         openmm_system = interchange.to_openmm()
#         openmm_topology = interchange.to_openmm_topology()
#         openmm_positions = interchange.positions.to_openmm()

#     return openmm_system, openmm_topology, openmm_positions

from parmed.tools.actions import changeRadii

def run_simulation(args):
    platform, properties = set_platform_properties(args)

    # logger.info("Preparing simulation...")
    # for filename in fixed_pdb_files:
    simulation, system, ligand_atom_indices, initial_atom_indices = set_simulation_parameters2(platform, properties, args)
    # simulation.context.reinitialize(True)
    logger.info('Minimizing complex energy...')
    minimize_and_save(simulation, args.outdir, args.name)
    logger.info("Performing simulation...")
    simulation.context.reinitialize(True)
    equilibriate(simulation, args, system)
    simulation.context.reinitialize(True)
    logger.info(f'Beginning simulation of {args.num_steps} steps with {args.step_size} fs step sizes...')
    # simulation.reporters.append(DCDReporter(args.output_traj_dcd, args.reporting_interval, enforcePeriodicBox=True))

    simulation.step(args.num_steps)
    logger.info(f'Finished simulation of {args.num_steps} steps!')

    structure = pmd.openmm.load_topology(
    simulation.topology,
    system,
    xyz=simulation.context.getState(positions=True).getPositions(asNumpy=True)
)
    changeRadii(structure, 'mbondi2').execute()
    structure.box = None
    prmtop_path = os.path.join(args.outdir, f"{args.name}_solvated_output.prmtop")
    inpcrd_path = os.path.join(args.outdir, f"{args.name}_solvated_output.inpcrd")

    structure.save(prmtop_path, format='amber', overwrite=True)
    structure = pmd.load_file(prmtop_path)
    structure.box = None
    structure.save(prmtop_path, format='amber', overwrite=True)
    nobox_prmtop_path = os.path.join(args.outdir, f"{args.name}_solvated_output_4mmgbsa.prmtop")
    nobox_comp_prmtop_path = os.path.join(args.outdir, f"{args.name}_complex_4mmgbsa.prmtop")
    sim_path = os.path.join(args.outdir, f'{args.name}_sim.pdb')
    sim_path_prmtop = os.path.join(args.outdir, f'{args.name}_sim.prmtop')

    write_nobox_prmtop(prmtop_path, nobox_prmtop_path)
    strip_waters_ions(prmtop_path, nobox_comp_prmtop_path)
    autoimage_center_image_trajout(nobox_prmtop_path, sim_path, sim_path_prmtop)

    mmpbsa_output_path = os.path.join(args.outdir, f'{args.name}_MMGBSA_output.dat')
    mmpbsa_input = write_mmpbsa_input_file(args)
    logger.info(f'Performing MMGBSA calculation on simulation...')
    run_mmpbsa(mmpbsa_input, nobox_prmtop_path, nobox_comp_prmtop_path, sim_path_prmtop, mmpbsa_output_path)
    gbsa_df = parse_generalized_born_section(mmpbsa_output_path)

    logger.info(f'Calculating RMSD/RMSF of simulation...')
    rms_output = compute_rms_metrics_protein_ligand_exclude_na_cl(nobox_comp_prmtop_path, args.sim_output_path, args.step_size)

    logger.info(f'Computing interaction fingerprint of simulation with ProLIF...')
    prolif_df = compute_interaction_fingerprint_postsim(prmtop_path, sim_path, ligand_atom_indices, initial_atom_indices, args.step_size)

    return gbsa_df, rms_output, prolif_df


def zero_out_atom_properties(mol_file, output_file):
    with open(mol_file, 'r') as f:
        lines = f.readlines()

    header = lines[:4]
    counts_line = lines[3]
    num_atoms = int(counts_line[:3])

    # Process atom block
    new_lines = header.copy()
    for i in range(4, 4 + num_atoms):
        parts = lines[i].split()
        if len(parts) < 4:
            new_lines.append(lines[i])  
            continue

        x, y, z, atom = parts[:4]
        # Reconstruct the line with zeroed properties (12 zeros, as standard)
        new_line = f"{float(x):10.4f}{float(y):10.4f}{float(z):10.4f} {atom:<3} {' 0'*12}\n"
        new_lines.append(new_line)

    # Append the rest of the file (bonds, etc.)
    new_lines.extend(lines[4 + num_atoms:])

    with open(output_file, 'w') as f:
        f.writelines(new_lines)

def set_simulation_parameters2(platform, properties, args):
    nonbonded_methods = {
    "NoCutoff": NoCutoff,
    "CutoffNonPeriodic": CutoffNonPeriodic,
    "CutoffPeriodic": CutoffPeriodic,
    "Ewald": Ewald,
    "PME": PME,
    "LJPME": LJPME  
        }
    # if args.martini_top:
    #     box_vec = [Vec3(args.periodic_box,0,0), Vec3(0,args.periodic_box,0), Vec3(0,0,args.periodic_box)]
    #     pdb = PDBFile(args.receptor)
    #     top = martini.MartiniTopFile(
	# 	args.martini_top,
	# 	periodicBoxVectors=box_vec,
	# 	defines={},
	# 	epsilon_r=15,
	# )
    #     system = top.create_system(nonbonded_cutoff=1.1 * nanometer)

    #     integrator = LangevinMiddleIntegrator(310 * kelvin,
	# 								10.0 / picosecond,
	# 								20 * femtosecond)

    #     simulation = Simulation(top.topology, system, integrator,
	# 						platform, properties)

    #     simulation.context.setPositions(pdb.positions)
    #     simulation.reporters.append(StateDataReporter(sys.stdout, 10, step=True, progress = True, totalSteps = (int(args.num_steps)+(2*args.equilibration_steps))))
    #     simulation.reporters.append(StateDataReporter(f'{args.name}_StateDataReporter.out', 10, step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, volume = True, density=True,  temperature=True, speed=True))
    #     simulation.reporters.append(PDBReporter(f'{args.name}_sim.pdb', 10000))
    #     return simulation, system



    pdb = PDBFile(args.protein)
    forcefield = ForceField(args.forcefield, args.solvent)
    if args.ligand:
        ligs = []
        if args.ligand.endswith('.pdb'):
            lig = PDBFile(args.ligand)
            ligs.append(lig)
        else:
            if args.multi:
                file_root, file_ext = os.path.splitext(args.ligand)
                if '_zeroed' in file_root:
                    pass
                else:
                    zeroed_out_path = file_root + "_zeroed" + file_ext
                zeroed_out_path = file_root + file_ext
                zero_out_atom_properties(args.ligand, zeroed_out_path)
                args.ligand = zeroed_out_path
            input_lig = Chem.SDMolSupplier(args.ligand)
            for mol in input_lig:
                lig = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
                smirnoff = SMIRNOFFTemplateGenerator(molecules=lig)
                forcefield.registerTemplateGenerator(smirnoff.generator)
                ligs.append(lig)
    modeller = Modeller(pdb.topology, pdb.positions)
    initial_atom_count = modeller.topology.getNumAtoms()
    initial_atom_indices = list(range(initial_atom_count))

    logger.info(f'System has added {args.protein} with {modeller.topology.getNumAtoms()} atoms')
    lig_positions = []
    lig_tops = []
    if args.ligand and not args.ligand.endswith('.pdb'):
        for lig in ligs:
            lig_topology = lig.to_topology()
            positions_array = lig_topology.get_positions().magnitude
            vec3_positions = [Vec3(x, y, z) for x, y, z in positions_array]
            final_positions = Quantity(vec3_positions, unit.nanometer)
            modeller.add(lig_topology.to_openmm(), final_positions)
            logger.info(f'System has added {lig} and with {lig.n_atoms} atoms')
    else:
        for lig in ligs:
            lig_topology = lig.getTopology()
            positions_array = lig.getPositions()
            vec3_positions = [Vec3(x, y, z) for x, y, z in positions_array]
            final_positions = Quantity(vec3_positions, unit.nanometer)
            modeller.add(lig_topology, final_positions)
            logger.info(f'System has added {lig} and with {lig_topology.getNumAtoms()} atoms')
    ligand_atom_indices = list(range(initial_atom_count, modeller.topology.getNumAtoms()))

    if 'tip' in args.solvent:
        args.solvent = args.solvent.split('/')[-1].split('.')[0]
        if args.solvent == 'tip3pfb':
            args.solvent = 'tip3p'
        modeller.addSolvent(forcefield, model=args.solvent, boxSize = Vec3(args.periodic_box, args.periodic_box, args.periodic_box))
    logger.info(f'System has {modeller.topology.getNumAtoms()} atoms after solvation')
    box_vec = [Vec3(args.periodic_box,0,0), Vec3(0,args.periodic_box,0), Vec3(0,0,args.periodic_box)]
    modeller.topology.setPeriodicBoxVectors(box_vec)
    if 'tip' in args.solvent:
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=nonbonded_methods[args.nonbonded_method], nonbondedCutoff=1*nanometer, constraints=args.constraints, rigidWater=args.rigidWater, flexibleConstraints=True)
    else:
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=nonbonded_methods[args.nonbonded_method], nonbondedCutoff=1*nanometer, constraints=args.constraints, rigidWater=args.rigidWater)
    logger.info(f'Periodic box size: x={args.periodic_box}, y={args.periodic_box}, z={args.periodic_box}')

    if args.apply_harmonic_force:
        harmonic_force = CustomExternalForce("0.5 * k * (z - z0)^2")
        harmonic_force.addGlobalParameter("k", args.force_constant)
        harmonic_force.addGlobalParameter("z0", args.z0)
        molecule_atom_indices = list(map(int, args.molecule_atom_indices.split(',')))
        for atom_index in molecule_atom_indices:
            harmonic_force.addParticle(atom_index, [])
        system.addForce(harmonic_force)

    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, args.step_size*femtosecond)
    simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)
    totalSteps = int(args.num_steps) + (int(args.equilibration_steps) * 2)
    simulation.reporters.append(StateDataReporter(os.path.join(args.outdir, f'{args.name}_StateDataReporter.out'), 100, step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, volume = True, density=True,  temperature=True, speed=True))
    simulation.reporters.append(PDBReporter(os.path.join(args.outdir, f'{args.name}_sim.pdb'), args.reporter_interval, enforcePeriodicBox=False))
    args.sim_output_path = os.path.join(args.outdir, f'{args.name}_sim.pdb')
    silent_output_stream = SilentOutputStream()
    progress_reporter = ProgressBarReporter(10, totalSteps, silent_output_stream, step=True)
    simulation.reporters.append(progress_reporter)
    return simulation, system, ligand_atom_indices, initial_atom_indices

def equilibriate(simulation, args, system):
    simulation.context.setVelocitiesToTemperature(100*kelvin)
    logger.info(f'Warming up temp to {300*kelvin}...')
    simulation.step(args.equilibration_steps)
    simulation.context.reinitialize(preserveState=True)
    logger.info(f'Pressurizing to {1*atmospheres}...')
    system.addForce(MonteCarloBarostat(1 * atmospheres, 300*kelvin, 25))
    # simulation.context.reinitialize(True)
    simulation.step(args.equilibration_steps)
    simulation.context.reinitialize(preserveState=True)

