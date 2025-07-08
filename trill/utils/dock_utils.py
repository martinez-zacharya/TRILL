import glob
import os
import shutil
import subprocess
import sys
from io import StringIO
import pkg_resources
from pathlib import Path
import pdbfixer
from Bio import SeqIO
from Bio.PDB import PDBParser, Superimposer, PDBIO
from biobb_vs.fpocket.fpocket_filter import fpocket_filter
from biobb_vs.fpocket.fpocket_run import fpocket_run
from biobb_vs.fpocket.fpocket_select import fpocket_select
from biobb_vs.utils.box import box
from openmm.app import element
import inspect
from openmm.app.pdbfile import PDBFile
from rdkit import Chem
import numpy as np
from tqdm import tqdm
import re
import requests
import MDAnalysis as mda
from icecream import ic
from rdkit.Chem import AllChem
from concurrent.futures import ThreadPoolExecutor, as_completed
from meeko import MoleculePreparation, RDKitMolCreate, PDBQTMolecule, PDBQTWriterLegacy
from loguru import logger
from os import linesep
import prolif as plf
# from .membrane_utils import insane, gro2pdb, prep4ldock


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

# Prepping input PDB file with protein and lipid bilayer for docking
# def prep4ldock(args):
#     cg_pdb_file_name = args.protein
#     output_pdb_file_name = args.protein.replace('.pdb', '_4ldock.pdb')

#     with open(cg_pdb_file_name) as ih:
#         with open(output_pdb_file_name, "w") as oh:
#             for line in ih:
#                 if line.startswith("ATOM  "):
#                     line = line.rstrip(linesep)
#                     if "PO4" in line:
#                         line = line.replace("PO4", " BJ").replace("DPPC", " MMB")
#                         oh.write(f"{line}{linesep}")
#                     else:
#                         res_name = line[12:16]
#                         if res_name in ["0BTN", "0BEN", "0BHN"] or res_name[0] == "B" or res_name[:2] == "5B":
#                             line = line.replace(res_name, "CA  ")
#                             oh.write(f"{line}{linesep}")
#     return output_pdb_file_name

def compute_interaction_fingerprint_postdock(receptor_pdb_path, ligand, ligand_pdbqts):
    _, ext = os.path.splitext(ligand)
    ext = ext.lower()
    if ext == '.mol' or ext == '.sdf':
        template = Chem.MolFromMolFile(ligand)
    elif ext == '.mol2':
        template = Chem.MolFromMol2File(ligand)
    u_receptor = mda.Universe(receptor_pdb_path)
    pose_iterable = plf.pdbqt_supplier(ligand_pdbqts, template)
    receptor_prolif = plf.Molecule.from_mda(u_receptor)
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
    fp_count = plf.Fingerprint(interactions, count=True, parameters={"VdWContact": {"preset": "rdkit"}})

    fp_count.run_from_iterable(pose_iterable, receptor_prolif)
    df = fp_count.to_dataframe(index_col="Pose")

    return df


def load_molecule(filename, removeHs=False):
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext == '.mol' or ext == '.sdf':
        return Chem.MolFromMolFile(filename, removeHs=removeHs)
    elif ext == '.pdb':
        return Chem.MolFromPDBFile(filename, removeHs=removeHs)
    elif ext == '.mol2':
        return Chem.MolFromMol2File(filename, removeHs=removeHs)
    else:
        raise ValueError(f'Unsupported file format: {ext}')
    
def parse_pocket_output(output):
    matched_pockets = []
    
    for line in output:
        if line.startswith("pocket"):
            matched_pockets.append(line)
        
    return matched_pockets

def convert_protein_to_pdbqt(protein_file, rec_pdbqt):
    logger.info(f'Converting {protein_file} to {rec_pdbqt}...')
    convert_rec = [
        'obabel',
        '-ipdb', protein_file,
        '--partialcharge', 'eem',
        '-h',
        '-xr',
        '-opdbqt',
        '-O', rec_pdbqt
    ]
    subprocess.run(convert_rec, stdout=subprocess.DEVNULL)

def convert_ligand_to_pdbqt(ligand_file, lig_pdbqt, lig_ext, args=None):
    logger.info(f'Converting {ligand_file} to {lig_pdbqt}...')
    if lig_ext == "pdb":
        convert_lig = [
            'obabel',
            '-ipdb',
            ligand_file,
            '-xr',
            '-h',
            '-- ', 'eem',
            '-opdbqt'
            '-O', lig_pdbqt
        ]
    else:
       with open(os.path.join(args.outdir, lig_pdbqt), 'w+') as outlig:
          input_lig = Chem.SDMolSupplier(ligand_file)
          for mol in input_lig:
            mol_Hs = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            AllChem.EmbedMolecule(mol_Hs, params)
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol_Hs)
            for setup in mol_setups:
                pdbqt_string = PDBQTWriterLegacy.write_string(setup)
                outlig.write(pdbqt_string[0])

    #     convert_lig = [
    #         'obabel',
    #         f'-i{lig_ext}',
    #         ligand_file,
    #         '-h',
    #         '--partialcharge',
    #         'eem',
    #         '-opdbqt',
    #         '-O', lig_pdbqt
    #     ]
    # subprocess.run(convert_lig, stdout=subprocess.DEVNULL)

def download_gnina(cache_dir):
    """
    Download gnina1.3.1 to the specified cache_dir as 'gnina' if not already present.

    Parameters:
        cache_dir (str): Path to the cache directory.

    Returns:
        str: Path to the downloaded 'gnina' file.
    """
    url = "https://github.com/gnina/gnina/releases/download/v1.3.1/gnina1.3.1"
    os.makedirs(cache_dir, exist_ok=True)
    gnina_path = os.path.join(cache_dir, "gnina")

    if not os.path.isfile(gnina_path):
        print(f"Downloading gnina to {gnina_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(gnina_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        os.chmod(gnina_path, 0o755)
        print("Download complete.")
    else:
        print("gnina already exists in cache.")

    return gnina_path

def extract_sequences(file_path):
    def extract_sequence_from_pdb(pdb_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
        sequence = ''
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  # Ensure it's a standard residue
                        sequence += SeqIO.Polypeptide.three_to_one(residue.resname)
        return sequence

    sequences = []
    
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            pdb_paths = file.readlines()
        for pdb_path in pdb_paths:
            pdb_path = pdb_path.strip()
            sequences.append(extract_sequence_from_pdb(pdb_path))
    elif file_path.endswith('.pdb'):
        sequences.append(extract_sequence_from_pdb(file_path))
    
    return sequences

def perform_docking(args, ligands):
    protein_file = os.path.abspath(args.protein)
    # ligand_file = os.path.abspath(ligands)
    
    if args.algorithm == 'LightDock':
        lightdock(args, ligands)
        return

    if args.algorithm == 'Gnina':
        download_gnina(args.cache_dir)
    # protein_name, prot_ext = os.path.basename(protein_file).split('.')
    protein_name = Path(protein_file).stem
    prot_ext = Path(protein_file).suffix
    # lig_name, lig_ext = os.path.basename(ligand_file).split('.')
    rec_pdbqt = f'{os.path.join(args.outdir, protein_name)}.pdbqt'
    # lig_pdbqt = f'{os.path.join(args.outdir, lig_name)}.pdbqt'

    # if prot_ext != 'pdbqt' and args.algorithm != 'Smina':
    convert_protein_to_pdbqt(protein_file, rec_pdbqt)

    if not args.blind and not args.multi_lig:
        run_fpocket_hunting(protein_file, protein_name, args)
        pockets = run_fpocket_filtering(protein_name)
    docking_results = []

    output_pdbqts = []
    if args.multi_lig:
      prepped_ligs = ''
      for current_lig in ligands:
        ligand_file = os.path.abspath(current_lig)
        lig_name, lig_ext = os.path.basename(ligand_file).split('.')
        lig_pdbqt = f'{os.path.join(args.outdir, lig_name)}.pdbqt'
        
        # if args.algorithm == 'Vina' and lig_ext != 'pdbqt':
        #     convert_ligand_to_pdbqt(ligand_file, lig_pdbqt, lig_ext, args)
        convert_ligand_to_pdbqt(ligand_file, lig_pdbqt, lig_ext, args)
        prepped_ligs += lig_pdbqt

      output_file = os.path.join(args.outdir, f"{args.name}_{args.algorithm}.pdbqt")
      args.output_file = output_file
      args.protein = rec_pdbqt
      args.ligand = prepped_ligs
      result = vina_dock(args, '', ligand_file)
      docking_results.append(('blind_dock:', result.stdout))
              
    else: 
      for current_lig in ligands:
          ligand_file = os.path.abspath(current_lig)
          lig_name, lig_ext = os.path.basename(ligand_file).split('.')
          lig_pdbqt = f'{os.path.join(args.outdir, lig_name)}.pdbqt'
        #   if args.algorithm == 'Vina' and lig_ext != 'pdbqt' and lig_ext != 'txt':
          if lig_ext != 'pdbqt' and lig_ext != 'txt':
              convert_ligand_to_pdbqt(ligand_file, lig_pdbqt, lig_ext, args)

          if not args.blind:
              for pocket in tqdm(pockets, desc=f"Docking {protein_name} and {lig_name}"):
                with Capturing() as output3:
                  fpocket_select(f'{protein_name}_filtered_pockets.zip', f'{protein_name}_filtered_pockets/{pocket}_atm.pdb', f'{protein_name}_filtered_pockets/{pocket}_vert.pqr')
                  prop = {
                  'offset': 2,
                  'box_coordinates': True
                  }
                  box(input_pdb_path=f'{protein_name}_filtered_pockets/{pocket}_vert.pqr',output_pdb_path=f'{protein_name}_filtered_pockets/{pocket}_box.pdb',properties=prop)

                pocket_file = f'{protein_name}_filtered_pockets/{pocket}_box.pdb'
                output_file = os.path.join(args.outdir, f"{lig_name}_{pocket}_{args.algorithm}.pdbqt")
                args.output_file = output_file
                output_pdbqts.append(output_file)
                if args.algorithm == 'Vina':
                  args.protein = rec_pdbqt
                  args.ligand = lig_pdbqt
                  ic(args.ligand)
                  ic(pocket_file)
                  ic(ligand_file)
                  result = vina_dock(args, pocket_file, ligand_file)
                  docking_results.append((pocket, result.stdout))
                elif args.algorithm == 'Smina':
                  args.protein = rec_pdbqt
                  args.ligand = lig_pdbqt
                  result = smina_dock(args, pocket_file, ligand_file)
                  docking_results.append((pocket, result.stdout))
                elif args.algorithm == 'Gnina':
                  args.protein = rec_pdbqt
                  args.ligand = lig_pdbqt
                  result = gnina_dock(args, pocket_file, ligand_file)
                  docking_results.append((pocket, result.stdout))

                for log_file in glob.glob("log*.out"):
                  os.remove(log_file)
                for log_file in glob.glob("log*.err"):
                  os.remove(log_file)
          else:
            output_file = os.path.join(args.outdir, f"{args.name}_{args.algorithm}.pdbqt")
            args.output_file = output_file
            if args.algorithm == 'Vina':
                args.protein = rec_pdbqt
                args.ligand = lig_pdbqt
                result = vina_dock(args, '', ligand_file)
                docking_results.append(('blind_dock:', result.stdout))
            elif args.algorithm == 'Smina':
                args.protein = rec_pdbqt
                args.ligand = lig_pdbqt
                result = smina_dock(args, '', ligand_file)
                docking_results.append(('blind_dock:', result.stdout))
            elif args.algorithm == 'Gnina':
                args.protein = rec_pdbqt
                args.ligand = lig_pdbqt
                result = gnina_dock(args, '', ligand_file)
                docking_results.append(('blind_dock:', result.stdout))
            output_pdbqts.append(output_file)

      for ix, output_pdbqt in enumerate(output_pdbqts):
        patt, num_poses = run_vina_split(output_pdbqt)
        lig_pdbqt_files = sorted(glob.glob(f"{patt}_pose*.pdbqt"))
        filtered_pose_files = []

        if num_poses == 0:
            filtered_pose_files.append(f"{patt}_pose1.pdbqt")
        else:
            for i in range(1, num_poses + 1):
                expected_file = f"{patt}_pose{i}.pdbqt"
                if expected_file in lig_pdbqt_files:
                    filtered_pose_files.append(expected_file)

        lig_pdbqt_files = sorted(filtered_pose_files)
        prolif_output_df = compute_interaction_fingerprint_postdock(protein_file, ligand_file, lig_pdbqt_files)
        if not prolif_output_df.empty:
            prolif_output_df.columns = prolif_output_df.columns.droplevel(0)

            prolif_output_df.columns = [f"{residue}|{interaction}" for residue, interaction in prolif_output_df.columns]
            prolif_output_df = prolif_output_df.reset_index()
            prolif_output_df['Pose'] = prolif_output_df['Pose'] + 1
            if args.blind:
                prolif_output_df.to_csv(os.path.join(args.outdir, f'{args.name}_{args.algorithm}_{lig_name}_ProLIF_output.csv'), index=None)
            else:
                prolif_output_df.to_csv(os.path.join(args.outdir, f'{args.name}_{args.algorithm}_{lig_name}_pocket{ix+1}_ProLIF_output.csv'), index=None)

    return docking_results

def write_docking_results_to_file(docking_results, args, protein_name, algorithm):
    with open(os.path.join(args.outdir, f'{args.name}_{algorithm}.out'), 'w+') as out:
        for num, res in docking_results:
            res_out = res.decode('utf-8')
            out.write(f'{protein_name}_{num}: \n')
            out.write(res_out)
            out.write('\n')
            out.write('-------------------------------------------------------------------------------------------------- \n')

def run_fpocket_hunting(protein_file, protein_name, args):
    prop = {
        'min_radius': args.min_radius,
        'max_radius': args.max_radius,
        'num_spheres': args.min_alpha_spheres,
        'sort_by': 'score'
    }
    logger.info('Pocket hunting...')
    with Capturing() as output1:
        fpocket_run(input_pdb_path=protein_file,
                    output_pockets_zip=f'{protein_name}_raw_pockets.zip',
                    output_summary=f'{protein_name}_fpocket_info.json',
                    properties=prop)
    return output1

def run_fpocket_filtering(protein_name):
    prop = {'score': [0.2, 1]}
    logger.info('Pocket filtering...')
    with Capturing() as output2:
        fpocket_filter(input_pockets_zip = f'{protein_name}_raw_pockets.zip', input_summary = f'{protein_name}_fpocket_info.json', output_filter_pockets_zip = f'{protein_name}_filtered_pockets.zip', properties=prop)
    pockets = parse_pocket_output(output2)
    shutil.unpack_archive(f'{protein_name}_filtered_pockets.zip', f'{protein_name}_filtered_pockets', 'zip')
    return pockets

def extract_box_info_from_pdb(pdb_file_path):
    with open(pdb_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if "REMARK BOX CENTER" in line:
            parts = line.split(":")
            center = tuple(map(float, parts[1].split("SIZE")[0].strip().split()))
            size = tuple(map(float, parts[2].strip().split()))
            return center, size
        
def create_init_file(dir_path):
    init_file = os.path.join(dir_path, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('# Automatically created __init__.py\n')
            
def smina_dock(args, pocket_file, ligand_file):
    # print(f"Smina docking with {args.protein} and {ligand_file} in pocket {pocket_file}")
    if args.blind:
      logger.info('Smina blind docking...')


      # Prepare the Smina command
      smina_cmd = [
          "smina",
          "-r", args.protein,
          "-l", args.ligand,
          "--autobox_ligand", args.protein,
          "--exhaustiveness", str(args.exhaustiveness),
          "-o", args.output_file
      ]
    else:
      # Extract center and size from the pocket PDB file
      center, size = extract_box_info_from_pdb(pocket_file)
      # Prepare the Smina command
      smina_cmd = [
          "smina",
          "-r", args.protein,
          "-l", args.ligand,
          "--center_x", str(center[0]),
          "--center_y", str(center[1]),
          "--center_z", str(center[2]),
          "--size_x", str(size[0]),
          "--size_y", str(size[1]),
          "--size_z", str(size[2]),
          "--minimize",
          "--exhaustiveness", str(args.exhaustiveness),
          "-o", args.output_file,
          "--local_only"
      ]

    # Run the Smina command
    result = subprocess.run(smina_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if not args.ligand.endswith('.txt'):
      logger.info(f"Docking completed. Results saved to {args.output_file}")
    return result

def gnina_dock(args, pocket_file, ligand_file):
    if args.blind:
      logger.info('Gnina blind docking...')

      # Prepare the Smina command
      gnina_cmd = [
          os.path.join(args.cache_dir, "gnina"),
          "-r", args.protein,
          "-l", args.ligand,
          "--autobox_ligand", args.protein,
          "--exhaustiveness", str(args.exhaustiveness),
          "-o", args.output_file
      ]
    else:
      # Extract center and size from the pocket PDB file
      center, size = extract_box_info_from_pdb(pocket_file)
      # Prepare the Smina command
      gnina_cmd = [
          os.path.join(args.cache_dir, "gnina"),
          "-r", args.protein,
          "-l", args.ligand,
          "--center_x", str(center[0]),
          "--center_y", str(center[1]),
          "--center_z", str(center[2]),
          "--size_x", str(size[0]),
          "--size_y", str(size[1]),
          "--size_z", str(size[2]),
          "--minimize",
          "--exhaustiveness", str(args.exhaustiveness),
          "-o", args.output_file,
          "--local_only",
          
      ]

    if int(args.GPUs) == 0:
        gnina_cmd.append('--no_gpu')
    ic(' '.join(gnina_cmd))
    # Run the Smina command
    result = subprocess.run(gnina_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if not args.ligand.endswith('.txt'):
      logger.info(f"Docking completed. Results saved to {args.output_file}")
    return result

def calculate_blind_bounding_box(pdbqt_file_path):
    """
    Calculate the center and size of the bounding box for a given .pdbqt file.

    Parameters:
    pdbqt_file_path (str): Path to the .pdbqt file.

    Returns:
    tuple: Center coordinates (center_x, center_y, center_z) and size (size_x, size_y, size_z).
    """
    with open(pdbqt_file_path, 'r') as file:
        lines = file.readlines()

    atom_lines = [line for line in lines if line.startswith('ATOM') or line.startswith('HETATM')]
    coordinates = []
    for line in atom_lines:
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        coordinates.append([x, y, z])

    # Convert to numpy array for easy calculations
    coords = np.array(coordinates)

    # Calculate the center and size of the bounding box
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords

    return center, size

def run_vina_split(pdbqt_file):
    if not os.path.isfile(pdbqt_file):
        raise FileNotFoundError(f"{pdbqt_file} not found.")

    # Extract base filename without extension
    filename = os.path.basename(pdbqt_file).split('.')[0]

    # Count poses by parsing MODEL lines
    max_model = 0
    with open(pdbqt_file, 'r') as f:
        for line in f:
            if line.startswith("MODEL"):
                model_num = int(line.strip().split()[1])
                if model_num > max_model:
                    max_model = model_num
    if max_model == 0:
        pdbqt_path = Path(pdbqt_file)
        new_name = pdbqt_path.with_name(pdbqt_path.stem + '_pose1' + pdbqt_path.suffix)
        shutil.copy(pdbqt_path, new_name)
        return filename, max_model
    else:
        cmd = ["vina_split", "--input", pdbqt_file, "--ligand", f"{filename}_pose"]
        subprocess.run(cmd, check=True)

        return filename, max_model

def convert_pdbqt_to_mol2(patt):
    pdbqt_files = sorted(glob.glob(f"{patt}_pose*.pdbqt"))
    if not pdbqt_files:
        raise RuntimeError(f"{patt}_pose_*.pdbqt files found after vina_split.")
    output_files = []
    for f in pdbqt_files:
        output_file = f.replace(".pdbqt", ".mol")
        cmd = ["obabel", f, "-O", output_file, "-h", "--gen-3d"]
        output_files.append(output_file)
        subprocess.run(cmd, check=True)
    return output_files

def vina_dock(args, pocket_file, ligand_file):
    split_paths = ' '.join(re.findall(r'(?:\./|/)[^\.]+\.pdbqt', args.ligand))
    
    if args.blind or args.multi_lig:
        logger.info('Vina blind docking...')
        center, size = calculate_blind_bounding_box(args.protein)
        vina_cmd = (
            f"vina --receptor {args.protein} --ligand {split_paths} "
            f"--center_x {center[0]} --center_y {center[1]} --center_z {center[2]} "
            f"--size_x {size[0]} --size_y {size[1]} --size_z {size[2]} "
            f"--exhaustiveness {args.exhaustiveness} --out {args.output_file}"
        ).split()
        result = subprocess.run(vina_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    else:
        # Extract center and size from the pocket PDB file
        center, size = extract_box_info_from_pdb(pocket_file)

        def try_docking(scale):
            logger.info(f"Attempting docking with size scale factor: {scale}")
            scaled_size = [s * scale for s in size]
            vina_cmd = [
                "vina",
                "--receptor", args.protein,
                "--ligand", args.ligand,
                "--center_x", str(center[0]),
                "--center_y", str(center[1]),
                "--center_z", str(center[2]),
                "--size_x", str(scaled_size[0]),
                "--size_y", str(scaled_size[1]),
                "--size_z", str(scaled_size[2]),
                "--exhaustiveness", str(args.exhaustiveness),
                "--out", args.output_file,
                "--local_only"
            ]
            return subprocess.run(vina_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        try:
            result = try_docking(1)
        except subprocess.CalledProcessError as e:
            logger.warning("Initial docking failed, trying with size * 5...")
            try:
                result = try_docking(5)
            except subprocess.CalledProcessError as e2:
                logger.warning("Retry with size * 5 failed, trying with size * 10...")
                try:
                    result = try_docking(10)
                except subprocess.CalledProcessError as e3:
                    logger.error("All retries failed. Vina could not complete docking.")
                    raise e3

    if not args.ligand.endswith('.txt'):
        logger.info(f"Docking completed. Results saved to {args.output_file}")
    
    return result

def calculate_rmsd(chain1, chain2):
    atoms1 = [atom for atom in chain1.get_atoms()]
    atoms2 = [atom for atom in chain2.get_atoms()]
    
    # Make sure the two lists have the same size
    if len(atoms1) != len(atoms2):
        return float('inf')
        
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    return sup.rms


def find_best_match(structure1, structure2):
    min_rmsd = float('inf')
    best_match = (None, None)
    
    for chain1 in structure1.get_chains():
        for chain2 in structure2.get_chains():
            rmsd = calculate_rmsd(chain1, chain2)
            
            if rmsd == float('inf'):
                continue
            
            if rmsd < min_rmsd:
                min_rmsd = rmsd
                best_match = (chain1, chain2)
                
    return best_match

def fixer_of_pdbs(args):
    fixed_pdb_files = []
    if args.just_relax:
        if len(args.receptor) > 1:
            if int(args.n_workers) > 1:
                with ThreadPoolExecutor(max_workers=int(args.n_workers)) as executor:
                    futures = {
                        executor.submit(fix_pdb, rec, {}, args): rec
                        for rec in args.receptor
                    }
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Fixing PDBs"):
                        rec = futures[future]
                        try:
                            fixed_pdb = future.result()
                            fixed_pdb_files.append(fixed_pdb)
                        except Exception as e:
                            print(f"Error processing receptor {rec}: {e}")
            else:
                for rec in tqdm(args.receptor, desc="Fixing PDBs"):
                    alterations = {}
                    fixed_pdb = fix_pdb(rec, alterations, args)
                    fixed_pdb_files.append(fixed_pdb)
        else:
            alterations = {}
            fixed_pdb = fix_pdb(args.receptor[0], alterations, args)
            fixed_pdb_files.append(fixed_pdb)   
    elif not args.ligand:
      receptor = fix_pdb(args.receptor, alterations={}, args=args)
      fixed_pdb_files.append(receptor)
    else:
      receptor = fix_pdb(args.receptor, alterations={}, args=args)
      if args.ligand.endswith(".pdb"):
        ligand = fix_pdb(args.ligand, alterations={}, args=args)
      else:
         ligand = args.ligand
      fixed_pdb_files.append(receptor)
      fixed_pdb_files.append(ligand)
    # elif args.structure:
    #     alterations = {}
    #     fixed_pdb = fix_pdb(args.structure, alterations, args)
    #     fixed_pdb_files.append(fixed_pdb)
    # elif args.dir:
    #     pdb_files = [f for f in os.listdir(args.dir) if f.endswith('.pdb')]
    #     with Pool(int(args.n_workers)) as p:
    #         fixed_pdb_files = list(tqdm(p.imap_unordered(partial(process_pdb, args), pdb_files), total=len(pdb_files), desc="Preprocessing PDBs..."))

    return fixed_pdb_files

def process_pdb(args, filename):
    alterations = {}
    fixed_pdb = fix_pdb(os.path.join(args.dir, filename), alterations, args)
    return fixed_pdb

def fix_pdb(pdb, alterations, args):
  """Apply pdbfixer to the contents of a PDB file; return a PDB string result.

  1) Replaces nonstandard residues.
  2) Removes heterogens (non protein residues) including water.
  3) Adds missing residues and missing atoms within existing residues.
  4) Adds hydrogens assuming pH=7.0.
  5) KeepIds is currently true, so the fixer must keep the existing chain and
     residue identifiers. This will fail for some files in wider PDB that have
     invalid IDs.

  Args:
    pdbfile: Input PDB file handle.

  Returns:
    A PDB string representing the fixed structure.
  """
  fixer = pdbfixer.PDBFixer(pdb)
  fixer.findNonstandardResidues()
  alterations['nonstandard_residues'] = fixer.nonstandardResidues
  fixer.replaceNonstandardResidues()
  # _remove_heterogens(fixer, alterations, keep_water=False)
  fixer.removeHeterogens(False)
  fixer.findMissingResidues()
  alterations['missing_residues'] = fixer.missingResidues
  fixer.findMissingAtoms()
  alterations['missing_heavy_atoms'] = fixer.missingAtoms
  alterations['missing_terminals'] = fixer.missingTerminals
  fixer.addMissingAtoms(seed=0)
  fixer.addMissingHydrogens()
  filename = os.path.splitext(os.path.basename(pdb))[0]
  out_file_path = os.path.join(args.outdir, f"{filename}_fixed.pdb")

  with open(out_file_path, 'w+') as f:
    PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

  return out_file_path


def clean_structure(pdb_structure, alterations_info):
  """Applies additional fixes to an OpenMM structure, to handle edge cases.

  Args:
    pdb_structure: An OpenMM structure to modify and fix.
    alterations_info: A dict that will store details of changes made.
  """
  _replace_met_se(pdb_structure, alterations_info)
  _remove_chains_of_length_one(pdb_structure, alterations_info)


def _remove_heterogens(fixer, alterations_info, keep_water):
  """Removes the residues that Pdbfixer considers to be heterogens.

  Args:
    fixer: A Pdbfixer instance.
    alterations_info: A dict that will store details of changes made.
    keep_water: If True, water (HOH) is not considered to be a heterogen.
  """
  initial_resnames = set()
  for chain in fixer.topology.chains():
    for residue in chain.residues():
      initial_resnames.add(residue.name)
  fixer.removeHeterogens(keepWater=keep_water)
  final_resnames = set()
  for chain in fixer.topology.chains():
    for residue in chain.residues():
      final_resnames.add(residue.name)
  alterations_info['removed_heterogens'] = (
      initial_resnames.difference(final_resnames))


def _replace_met_se(pdb_structure, alterations_info):
  """Replace the Se in any MET residues that were not marked as modified."""
  modified_met_residues = []
  for res in pdb_structure.iter_residues():
    name = res.get_name_with_spaces().strip()
    if name == 'MET':
      s_atom = res.get_atom('SD')
      if s_atom.element_symbol == 'Se':
        s_atom.element_symbol = 'S'
        s_atom.element = element.get_by_symbol('S')
        modified_met_residues.append(s_atom.residue_number)
  alterations_info['Se_in_MET'] = modified_met_residues


def _remove_chains_of_length_one(pdb_structure, alterations_info):
  """Removes chains that correspond to a single amino acid.

  A single amino acid in a chain is both N and C terminus. There is no force
  template for this case.

  Args:
    pdb_structure: An OpenMM pdb_structure to modify and fix.
    alterations_info: A dict that will store details of changes made.
  """
  removed_chains = {}
  for model in pdb_structure.iter_models():
    valid_chains = [c for c in model.iter_chains() if len(c) > 1]
    invalid_chain_ids = [c.chain_id for c in model.iter_chains() if len(c) <= 1]
    model.chains = valid_chains
    for chain_id in invalid_chain_ids:
      model.chains_by_id.pop(chain_id)
    removed_chains[model.number] = invalid_chain_ids
  alterations_info['removed_chains'] = removed_chains


def get_structure(file_path):
    parser = PDBParser()
    return parser.get_structure(file_path.split('/')[-1].split('.')[0], file_path)

def find_matching_chain(structure1, structure2):
    for chain1 in structure1.get_chains():
        for chain2 in structure2.get_chains():
            atoms1 = [atom.id for atom in chain1.get_atoms()]
            atoms2 = [atom.id for atom in chain2.get_atoms()]
            if set(atoms1) == set(atoms2):
                return chain1, chain2
    return None, None

def superimpose_chains(chain1, chain2):
    sup = Superimposer()
    atoms1 = [atom for atom in chain1.get_atoms()]
    atoms2 = [atom for atom in chain2.get_atoms()]
    sup.set_atoms(atoms1, atoms2)
    sup.apply(chain1.get_atoms())

def save_complex(receptor_structure, ligand_structure, output_file):
    io = PDBIO()
    io.set_structure(receptor_structure)
    io.save(output_file)
    io.set_structure(ligand_structure)
    io.save(output_file, write_end=1, append_end=1)

def downgrade_biopython():
    """ Downgrade biopython to version 1.7.9 """
    # Check if we're in a pixi environment
    if os.path.exists(os.path.join(os.path.dirname(sys.executable), '..', '..', 'pixi.toml')):
        print("Warning: Running in pixi environment. Skipping package downgrade for LightDock compatibility.")
        print("LightDock may not work properly with current package versions.")
        return
    
    # Try pip first
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'biopython==1.79', 'numpy==1.23.5', 'pyparsing==3.1.1'])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Could not downgrade packages. LightDock may not work properly.")
        print("Please ensure biopython==1.79, numpy==1.23.5, and pyparsing==3.1.1 are installed.")

def upgrade_biopython(og_ver_biopython, og_ver_np, pypar_ver):
    """ Upgrade biopython back to the original version """
    # Check if we're in a pixi environment
    if os.path.exists(os.path.join(os.path.dirname(sys.executable), '..', '..', 'pixi.toml')):
        print("Skipping package upgrade in pixi environment.")
        return
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'biopython=={og_ver_biopython}', f'numpy=={og_ver_np}', f'pyparsing=={pypar_ver}'])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Could not upgrade packages back to original versions.")

def get_current_biopython_version():
    """ Get the current version of biopython """
    biopy_ver = pkg_resources.get_distribution("biopython").version
    np_ver = pkg_resources.get_distribution("numpy").version
    pypar_ver = pkg_resources.get_distribution("pyparsing").version

    return biopy_ver, np_ver, pypar_ver


def lightdock(args, ligands):
    og_biopython_ver, og_np_ver, pypar_ver = get_current_biopython_version()
    downgrade_biopython()
    master_outdir = os.path.abspath(args.outdir) # Save the original output directory
    protein_file = os.path.basename(args.protein)
    args.protein = os.path.abspath(args.protein)
    if len(ligands) > 1:
        for ligand_path in ligands:
            ligand_name = os.path.splitext(os.path.basename(ligand_path))[0]
            args.ligand = os.path.abspath(ligand_path)  # Make sure it's an absolute path
            args.outdir = os.path.join(master_outdir, ligand_name)  # Create a sub-directory for each ligand

            # Create the sub-directory if it doesn't exist
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)

            prot_dest = os.path.join(args.outdir, os.path.basename(args.protein))
            lig_dest = os.path.join(args.outdir, os.path.basename(args.ligand))
            if os.path.abspath(args.protein) != os.path.abspath(prot_dest):
              shutil.copy(args.protein, args.outdir)
            if os.path.abspath(args.ligand) != os.path.abspath(lig_dest):
              shutil.copy(args.ligand, args.outdir) 

            args.protein = os.path.join(args.outdir, protein_file) 
            args.ligand = os.path.join(args.outdir, os.path.basename(args.ligand))  
            # Run the LightDock pipeline for this ligand
            lightdock_out = os.path.join(args.outdir, f'{args.name}_lightdock_output')
            os.mkdir(lightdock_out)
            args.outdir = lightdock_out
            fixed_prot = fix_pdb(args.protein, {}, args)
            args.protein = fixed_prot
            fixed_ligand = fix_pdb(args.ligand, {}, args)
            args.ligand = fixed_ligand  
            lightdock_setup(args)
            lightdock_run(os.path.join(args.outdir, 'setup.json'), args.sim_steps, args.outdir, args.n_workers)
            generate_ant_thony_list(args)
            rank_lightdock(args)
            generate_lightdock_conformations(args)
            cluster_lightdock_conformations(args)
    else:  # If there's only one ligand, proceed as usual
        ligand_file = os.path.basename(args.ligand)
        prot_dest = os.path.join(args.outdir, os.path.basename(args.protein))
        lig_dest = os.path.join(args.outdir, os.path.basename(args.ligand))
        # Check if source and destination are the same
        if os.path.abspath(args.protein) != os.path.abspath(prot_dest):
            shutil.copy(args.protein, args.outdir)
        if os.path.abspath(args.ligand) != os.path.abspath(lig_dest):
            shutil.copy(args.ligand, args.outdir)  

        args.protein = os.path.join(args.outdir, protein_file)
        args.ligand = os.path.join(args.outdir, ligand_file)
        fixed_prot = fix_pdb(args.protein, {}, args)
        args.protein = fixed_prot
        # if not args.membrane:
        #     fixed_prot = fix_pdb(args.protein, {}, args)
        #     args.protein = fixed_prot
        # else:
        #     insane(args)
        #     # gro2pdb(args)
        #     prep4ldock(args)
        # # args.protein = prep4ldock(args)
        fixed_ligand = fix_pdb(args.ligand, {}, args)
        args.ligand = fixed_ligand  
        lightdock_setup(args)
        lightdock_run(os.path.join(args.outdir, 'setup.json'), args.sim_steps, args.outdir, args.n_workers)
        generate_ant_thony_list(args)
        rank_lightdock(args)
        generate_lightdock_conformations(args)
        cluster_lightdock_conformations(args)
    upgrade_biopython(og_biopython_ver, og_np_ver, pypar_ver)

def lightdock_setup(args):
    if args.restraints:
      cmd = ["lightdock3_setup.py", args.protein, args.ligand, "--outdir", args.outdir, "--noxt", "--noh", "--now", "-s", str(args.swarms), "--seed_points", str(args.RNG_seed), "--seed_anm", str(args.RNG_seed), "--rst", args.restraints, "-g", str(args.glowworms)]
    elif args.membrane:
        cmd = ["lightdock3_setup.py", args.protein, args.ligand, "--outdir", args.outdir, "--noxt", "--noh", "--now", "--membrane", "-s", str(args.swarms), "--seed_points", str(args.RNG_seed), "--seed_anm", str(args.RNG_seed), "-g", str(args.glowworms)]
    else:
      cmd = ["lightdock3_setup.py", args.protein, args.ligand,"--outdir", args.outdir, "--noxt", "--noh", "--now", "-s", str(args.swarms), "--seed_points", str(args.RNG_seed), "--seed_anm", str(args.RNG_seed),"-g", str(args.glowworms)]
    subprocess.run(cmd)

def lightdock_run(setup_json, steps, outdir, cpus):
    cmd = ["lightdock3.py", setup_json, str(steps), '--outdir', outdir, '-c', str(cpus)]
    subprocess.run(cmd)

def generate_ant_thony_list(args):
    with open(f"{args.outdir}/generate_lightdock.list", "w") as f:
        for i in range(args.swarms):
            f.write(f"cd {args.outdir}/swarm_{i}; lgd_generate_conformations.py {args.protein} {args.ligand} gso_{args.sim_steps}.out {args.sim_steps} > /dev/null 2> /dev/null;\n")

    with open(f"{args.outdir}/cluster_lightdock.list", "w") as f:
        for i in range(args.swarms):
            f.write(f"cd {args.outdir}/swarm_{i}; lgd_cluster_bsas.py gso_{args.sim_steps}.out > /dev/null 2> /dev/null;\n")

    cmd = ["ant_thony.py", "-c", str(args.n_workers), f"{args.outdir}/generate_lightdock.list"]
    subprocess.run(cmd)

    cmd = ["ant_thony.py", "-c", str(args.n_workers), f"{args.outdir}/cluster_lightdock.list"]
    subprocess.run(cmd)

def rank_lightdock(args):
    cmd = ["lgd_rank.py", str(args.swarms), str(args.sim_steps), "--outdir", args.outdir]
    subprocess.run(cmd)

def generate_lightdock_conformations(args):
    for i in range(args.swarms):
        swarm_dir = os.path.join(args.outdir, f"swarm_{i}")
        if not os.path.exists(swarm_dir):
            print(f"Directory {swarm_dir} does not exist. Skipping.")
            continue

        gso_file = os.path.join(swarm_dir, f"gso_{args.sim_steps}.out")
        if not os.path.exists(gso_file):
            print(f"GSO file {gso_file} does not exist in {swarm_dir}. Skipping.")
            continue

        cmd = [
            "lgd_generate_conformations.py",
            f"{args.protein}",
            f"{args.ligand}",
            gso_file,
            str(args.glowworms)
        ]
        subprocess.run(cmd)

def cluster_lightdock_conformations(args):
    for i in range(args.swarms):
        swarm_dir = os.path.join(args.outdir, f"swarm_{i}")
        if not os.path.exists(swarm_dir):
            print(f"Directory {swarm_dir} does not exist. Skipping.")
            continue

        gso_file = os.path.join(swarm_dir, f"gso_{args.sim_steps}.out")
        if not os.path.exists(gso_file):
            print(f"GSO file {gso_file} does not exist in {swarm_dir}. Skipping.")
            continue

        cmd = [
            "lgd_cluster_bsas.py",
            gso_file
        ]
        subprocess.run(cmd)