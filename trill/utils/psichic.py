# Utils for PSICHIC

import json
import pandas as pd
import torch
import numpy as np
import os
import random
import subprocess
from Bio import SeqIO
from itertools import product
import sys
from loguru import logger

def clone_and_install_psichic(cache_dir):
    repo_url = "https://github.com/martinez-zacharya/PSICHIC"
    target_path = os.path.join(cache_dir, "PSICHIC")

    if os.path.exists(target_path):
        logger.info(f"PSICHIC already exists at {target_path}. Skipping clone.")
    else:
        try:
            logger.info(f"Cloning PSICHIC into {target_path}...")
            subprocess.run(["git", "clone", repo_url, target_path], check=True)
            logger.info("Clone complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cloning PSICHIC: {e}")
            raise

    return target_path

def fasta_smiles_to_dataframe(fasta_path, smiles_path):
    # Parse FASTA sequences
    sequences = [(record.id, str(record.seq)) for record in SeqIO.parse(fasta_path, "fasta")]

    # Parse SMILES entries
    smiles_entries = []
    with open(smiles_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        for i in range(0, len(lines), 2):
            if lines[i].startswith(">") and (i + 1) < len(lines):
                smiles_string = lines[i + 1]
                smiles_entries.append(smiles_string)

    # Create rows as combinations of sequence and each SMILES string
    rows = []
    for seq_id, seq in sequences:
        for smiles in smiles_entries:
            rows.append({
                "ID": seq_id,
                "Protein": seq,
                "Ligand": smiles
            })

    return pd.DataFrame(rows)

def run_psichic(args, cache_dir):
    from utils.utils import DataLoader, virtual_screening
    from utils.dataset import ProteinMoleculeDataset
    from utils.trainer import Trainer
    # from utils.metrics import *
    from utils import protein_init, ligand_init
    from models.net import net
    from tqdm import tqdm
    PSICHIC_parameters = "Pre-trained on Large-scale Interaction Database" 
    Save_PSICHIC_Interpretation = False
    batch_size = 1
    trained_model_path = f'{cache_dir}/PSICHIC/trained_weights/multitask_PSICHIC'

    # This is the input csv for processing
    screenfile = args.input_csv
    result_path = args.outdir

    device = "cuda" if int(args.GPUs) >= 1 else "cpu"
    with open(os.path.join(trained_model_path,'config.json'),'r') as f:
        config = json.load(f)

    # device
    device = torch.device(device)


    if not os.path.exists(result_path):
        os.makedirs(result_path)

    degree_dict = torch.load(os.path.join(trained_model_path,'degree.pt'))
    param_dict = os.path.join(trained_model_path,'model.pt')
    mol_deg, prot_deg = degree_dict['ligand_deg'],degree_dict['protein_deg']

    model = net(mol_deg, prot_deg,
                # MOLECULE
                mol_in_channels=config['params']['mol_in_channels'],  prot_in_channels=config['params']['prot_in_channels'],
                prot_evo_channels=config['params']['prot_evo_channels'],
                hidden_channels=config['params']['hidden_channels'], pre_layers=config['params']['pre_layers'],
                post_layers=config['params']['post_layers'],aggregators=config['params']['aggregators'],
     scalers=config['params']['scalers'],total_layer=config['params']['total_layer'],
                K = config['params']['K'],heads=config['params']['heads'],
                dropout=config['params']['dropout'],
                dropout_attn_score=config['params']['dropout_attn_score'],
                # output
                regression_head=config['tasks']['regression_task'],
                classification_head=config['tasks']['classification_task'] ,
                multiclassification_head=config['tasks']['mclassification_task'],
                device=device).to(device)
    model.reset_parameters()
    model.load_state_dict(torch.load(param_dict,map_location=device))


    screen_df = pd.read_csv(os.path.join(screenfile))
    protein_seqs = screen_df['Protein'].unique().tolist()
    logger.info('Initialising protein sequence to Protein Graph')
    protein_dict = protein_init(protein_seqs, args)
    ligand_smiles = screen_df['Ligand'].unique().tolist()
    logger.info('Initialising ligand SMILES to Ligand Graph')
    ligand_dict = ligand_init(ligand_smiles)
    torch.cuda.empty_cache()
    ## drop any invalid smiles
    screen_df = screen_df[screen_df['Ligand'].isin(list(ligand_dict.keys()))].reset_index(drop=True)
    screen_dataset = ProteinMoleculeDataset(screen_df, ligand_dict, protein_dict, device=device)
    screen_loader = DataLoader(screen_dataset, batch_size=batch_size, shuffle=False,
                                follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])

    logger.info("Screening starts now!")
    screen_df = virtual_screening(screen_df, model, screen_loader,
                    result_path=os.path.join(result_path, "interpretation_result"), save_interpret=Save_PSICHIC_Interpretation,
                    ligand_dict=ligand_dict, device=device)

    screen_df.to_csv(os.path.join(result_path,f'{args.name}_PSICHIC_screening_output.csv'),index=False)
    logger.success('Screening completed and saved to {}'.format(result_path))