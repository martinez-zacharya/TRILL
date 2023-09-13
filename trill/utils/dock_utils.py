import os
import subprocess
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pandas as pd


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
    
def perform_docking(args):
    protein_file = os.path.abspath(args.protein)
    ligand_file = os.path.abspath(args.ligand)


    protein_name = os.path.basename(protein_file).split('.')[0]
    lig_name = os.path.basename(ligand_file).split('.')[0]

    # Load the ligand with RDKit
    ligand = load_molecule(ligand_file, removeHs=False)
    rec_pdbqt = f'{os.path.join(args.outdir, protein_name)}.pdbqt'
    lig_pdbqt = f'{os.path.join(args.outdir, lig_name)}.pdbqt'

    convert_rec = f'obabel {protein_file} -O {rec_pdbqt}'.split(' ')
    subprocess.run(convert_rec, stdout=subprocess.DEVNULL)
    convert_lig = f'obabel {ligand_file} -O {lig_pdbqt}'.split(' ')
    subprocess.run(convert_lig, stdout=subprocess.DEVNULL)

    # Calculate properties with RDKit
    mw = Descriptors.MolWt(ligand)
    logp = Descriptors.MolLogP(ligand)
    num_h_donors = rdkit.Chem.rdMolDescriptors.CalcNumHBD(ligand)
    num_h_acceptors = rdkit.Chem.rdMolDescriptors.CalcNumHBA(ligand)
    num_rotatable_bonds = rdkit.Chem.rdMolDescriptors.CalcNumRotatableBonds(ligand)
    tpsa = rdkit.Chem.rdMolDescriptors.CalcTPSA(ligand)  # Topological Polar Surface Area
    # Determine if the ligand is a small molecule or a protein based on its size
    if args.force_ligand:
        is_small_molecule = True if args.force_ligand == 'small' else False
    else:
        is_small_molecule = mw < 800  # Lowered the threshold
    if is_small_molecule:
        fpock_cmd = f"fpocket -f {protein_file}".split(' ')
        subprocess.run(fpock_cmd)
    else:
        fpock_cmd = f"fpocket -f {protein_file} -m 3.5 -M 10.0 -i 3 -n 2".split(' ')
        subprocess.run(fpock_cmd)

    # Load fpocket output
    output_file_path = f"{os.path.dirname(protein_file)}/{protein_name}_out/{protein_name}_info.txt"
    output_dir_path = f"{os.path.dirname(protein_file)}/{protein_name}_out"
    with open(output_file_path, 'r') as f:
        pockets_txt = f.read()

    # Split into individual pockets
    pockets_list = pockets_txt.split('Pocket ')[1:]
    # For each pocket in the filtered pockets, perform docking with smina
    lig_type = "small molecule" if is_small_molecule else "protein"
    print(f'Docking the ligand as a {lig_type}. If TRILL has assumed the type of ligand wrong, you can try docking again with the flag --force_ligand')
    docking_results = []
    for pocket_str in pockets_list:
        pocket_lines = pocket_str.split('\n')
        pocket_num = int(pocket_lines[0].strip(':'))
        pocket = {line.split(':')[0].strip(): float(line.split(':')[1].strip()) for line in pocket_lines[1:] if ':' in line}
        # Filter pockets based on ligand properties and pocket properties
        if is_small_molecule:
            if (pocket['Score'] > 0.2):
                # pocket['Druggability Score'] > 0.15 and 
                # pocket['Volume'] < mw*3):
                # pocket['Polarity score'] >= tpsa*0.8 and
                # pocket['Mean local hydrophobic density'] <= logp):
              
                pocket_file = f"{output_dir_path}/pockets/pocket{pocket_num}_atm.pdb"
                output_file = os.path.join(args.outdir, f"{lig_name}_pocket{pocket_num}.pdbqt")
                smina_cmd = f"smina -r {rec_pdbqt} -l {lig_pdbqt} --autobox_ligand {pocket_file} -o {output_file} --minimize".split(' ')
                result = subprocess.run(smina_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                docking_results.append((pocket_num, result.stdout))

        else:  # For proteins
            if (pocket['Score'] > 0.2):
                # pocket['Druggability Score'] > 0.4 and
                # pocket['Volume'] > mw*5):
                # pocket['Polarity score'] >= tpsa*1.2 and  
                # pocket['Mean local hydrophobic density'] <= logp*1.2):  
                pocket_file = f"{output_dir_path}/pockets/pocket{pocket_num}_atm.pdb"
                output_file = os.path.join(args.outdir, f"{lig_name}_pocket{pocket_num}.pdbqt")

                smina_cmd = f"smina -r {rec_pdbqt} -l {lig_pdbqt} --autobox_ligand {pocket_file} -o {output_file} --minimize".split(' ')
                result = subprocess.run(smina_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                docking_results.append((pocket_num, result.stdout))


    
    return docking_results




