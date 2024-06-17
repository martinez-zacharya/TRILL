import os
import sys
import time

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
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmmforcefields.generators import SystemGenerator
from openff.interchange import Interchange
from openff.toolkit import ForceField as ff_ff
from openff.toolkit.utils import get_data_file_path
from openmm.unit.quantity import Quantity
from tqdm import tqdm
from icecream import ic
from loguru import logger
import numpy as np
from openff.interchange import Interchange
from openmm import app
from rdkit import Chem
from openff.interchange.components._packmol import RHOMBIC_DODECAHEDRON, pack_box
from .cuda_utils import set_platform_properties


# import martini_openmm as martini

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


def run_simulation(args):
    platform, properties = set_platform_properties(args)

    logger.info("Preparing simulation...")
    # for filename in fixed_pdb_files:
    simulation, system = set_simulation_parameters2(platform, properties, args)
    # simulation.context.reinitialize(True)
    logger.info('Minimizing complex energy...')
    minimize_and_save(simulation, args.outdir, args.name)
    logger.info("Performing simulation...")
    simulation.context.reinitialize(True)
    equilibriate(simulation, args, system)
    simulation.context.reinitialize(True)
    logger.info(f'Beginning simulation of {args.num_steps} steps with {args.step_size} fs step sizes')
    # simulation.reporters.append(DCDReporter(args.output_traj_dcd, args.reporting_interval, enforcePeriodicBox=True))
    simulation.step(args.num_steps)

def set_simulation_parameters2(platform, properties, args):
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
            input_lig = Chem.SDMolSupplier(args.ligand)
            for mol in input_lig:
                lig = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
                smirnoff = SMIRNOFFTemplateGenerator(molecules=lig)
                forcefield.registerTemplateGenerator(smirnoff.generator)
                ligs.append(lig)
    modeller = Modeller(pdb.topology, pdb.positions)
    logger.info(f'System has added {args.protein} with {modeller.topology.getNumAtoms()} atoms')
    if args.ligand:
        for lig in ligs:
            lig_topology = lig.to_topology()
            positions_array = lig_topology.get_positions().magnitude
            vec3_positions = [Vec3(x, y, z) for x, y, z in positions_array]
            final_positions = Quantity(vec3_positions, unit.nanometer)
            modeller.add(lig_topology.to_openmm(), final_positions)
            logger.info(f'System has added {lig} and with {lig.n_atoms} atoms')
    if 'tip' in args.solvent:
        args.solvent = args.solvent.split('/')[-1].split('.')[0]
        if args.solvent == 'tip3pfb':
            args.solvent = 'tip3p'
        modeller.addSolvent(forcefield, model=args.solvent, boxSize = Vec3(args.periodic_box, args.periodic_box, args.periodic_box))
    logger.info(f'System has {modeller.topology.getNumAtoms()} atoms after solvation')
    nonbonded_methods = {
        "NoCutoff": NoCutoff,
        "CutoffNonPeriodic": CutoffNonPeriodic,
        "CutoffPeriodic": CutoffPeriodic,
        "Ewald": Ewald,
        "PME": PME,
        "LJPME": LJPME  
            }
    box_vec = [Vec3(args.periodic_box,0,0), Vec3(0,args.periodic_box,0), Vec3(0,0,args.periodic_box)]
    modeller.topology.setPeriodicBoxVectors(box_vec)
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
    totalSteps = int(args.num_steps)
    simulation.reporters.append(StateDataReporter(os.path.join(args.outdir, f'{args.name}_StateDataReporter.out'), 100, step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, volume = True, density=True,  temperature=True, speed=True))
    simulation.reporters.append(PDBReporter(os.path.join(args.outdir, f'{args.name}_sim.pdb'), args.reporter_interval, enforcePeriodicBox=False))
    silent_output_stream = SilentOutputStream()
    progress_reporter = ProgressBarReporter(10, totalSteps, silent_output_stream, step=True)
    simulation.reporters.append(progress_reporter)
    return simulation, system

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

