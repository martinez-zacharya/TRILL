# Utils for CatPred
import os
import requests
import tarfile
import subprocess
from itertools import product
from Bio import SeqIO
from rdkit import Chem
import sys
import pkg_resources
import pandas as pd
import numpy as np
from loguru import logger

def downgrade_rotary_emb():
    """Downgrade rotary_embedding_torch to version 0.6.5 silently"""
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'rotary_embedding_torch==0.6.5'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def upgrade_rotary_emb(og_ver_rotary):
    """Upgrade rotary_embedding_torch back to the original version silently"""
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', f'rotary_embedding_torch=={og_ver_rotary}'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def get_rotary_emb_version():
    """ Get the current version of rotary_embedding_torch """
    rotary_ver = pkg_resources.get_distribution("rotary_embedding_torch").version

    return rotary_ver

def tupulize_fasta_smiles(args):
    fasta_path = args.query
    smiles_path = args.smiles

    # Parse FASTA sequences
    sequences = [(record.id, str(record.seq)) for record in SeqIO.parse(fasta_path, "fasta")]

    # Parse SMILES "FASTA-like" entries
    smiles_entries = []
    with open(smiles_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        for i in range(0, len(lines), 2):
            if lines[i].startswith(">") and (i + 1) < len(lines):
                smiles_name = lines[i][1:]  # remove '>'
                smiles_string = lines[i + 1]
                smiles_entries.append((smiles_name, smiles_string))

    # Create 4-tuples of (sequence_id, sequence, smiles_name, smiles_string)
    result = [(seq_id, seq, smiles_name, smiles) for (seq_id, seq), (smiles_name, smiles) in product(sequences, smiles_entries)]

    return result
def create_csv_sh(parameter, uni, seq, smi, sminame, args):
    """
    Create input CSV and shell script for prediction.

    Args:
        parameter (str): Kinetics parameter to predict.
        uni (str): UniProt ID or enzyme name.
        seq (str): Enzyme sequence.
        smi (str): Substrate SMILES.

    Returns:
        tuple: Enzyme sequence and SMILES if successful, None otherwise.
    """
    mol = Chem.MolFromSmiles(smi)
    smi = Chem.MolToSmiles(mol)


    # Validate enzyme sequence
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    if not set(seq).issubset(valid_aas):
        logger.warn('Invalid Enzyme sequence input!')
        logger.warn('Correct your input! Exiting..')
        return None

    if parameter == 'kcat' and '.' in smi:
        smi = '.'.join(sorted(smi.split('.')))

    csv_path = os.path.join(args.outdir, f'{uni}_{parameter}_{args.name}_input.csv')
    with open(csv_path, 'w') as f:
        f.write('name,sequence,SMILES,pdbpath\n')
        f.write(f'{uni},{seq},{smi},{uni}.pdb\n')

    sh_path = os.path.join(args.outdir, f'{uni}_{parameter}_catpred_predict.sh')
    with open(sh_path, 'w') as f:
        f.write(f'''#!/bin/bash
TEST_FILE_PREFIX={uni}_{parameter}_{sminame}
OUT_FILE=${{TEST_FILE_PREFIX}}.json.gz
DATA_FILE_PREFIX={uni}_{parameter}
DATA_FILE=${{DATA_FILE_PREFIX}}_{args.name}_input.csv
OUTPUT_FILE=${{DATA_FILE_PREFIX}}_output.csv
CHECKPOINT_DIR={args.cache_dir}/CatPred_weights/pretrained/production/{parameter}/

python3 {args.cache_dir}/catpred/scripts/create_pdbrecords.py --data_file ${{DATA_FILE}} --out_file ${{OUT_FILE}}
python3 {args.cache_dir}/catpred/predict.py --test_path ${{DATA_FILE}} --checkpoint_dir ${{CHECKPOINT_DIR}} --uncertainty_method mve --smiles_column SMILES --individual_ensemble_predictions --preds_path ${{OUTPUT_FILE}} --protein_records_path ${{OUT_FILE}}
''')

    logger.info('Input success!')
    logger.info(f'{uni} sequence length: {len(seq)}')
    logger.info(f'{sminame} structure: {smi}')

    return seq, smi, sh_path

def get_predictions(parameter, uniprot_id):
    """
    Process and display prediction results.

    Args:
        parameter (str): Kinetics parameter that was predicted.
        uniprot_id (str): UniProt ID or enzyme name used in prediction.
    """
    df = pd.read_csv(f'{uniprot_id}_{parameter}_output.csv')

    # Set parameter-specific variables
    if parameter == 'kcat':
        parameter_print, parameter_print_log = 'k_{cat}', 'log_{10}(k_{cat})'
        target_col, unit = 'log10kcat_max', ' s^{-1}'
    elif parameter == 'km':
        parameter_print, parameter_print_log = 'K_{m}', 'log_{10}(K_{m})'
        target_col, unit = 'log10km_mean', ' mM'
    else:
        parameter_print, parameter_print_log = 'K_{i}', 'log_{10}(K_{i})'
        target_col, unit = 'log10ki_mean', ' mM'

    unc_col = f'{target_col}_mve_uncal_var'
    model_cols = [col for col in df.columns if col.startswith(target_col) and 'model_' in col]

    # Extract predictions and calculate uncertainties
    unc = df[unc_col].iloc[0]
    prediction = df[target_col].iloc[0]
    prediction_linear = np.power(10, prediction)
    model_outs = np.array([df[col].iloc[0] for col in model_cols])
    epi_unc = np.var(model_outs)
    alea_unc = unc - epi_unc
    epi_unc, alea_unc, unc = np.sqrt(epi_unc), np.sqrt(alea_unc), np.sqrt(unc)

    return unc, epi_unc, alea_unc, prediction, prediction_linear

def clone_and_install_catpred(cache_dir):
    repo_url = "https://github.com/martinez-zacharya/catpred"
    target_path = os.path.join(cache_dir, "catpred")

    if os.path.exists(target_path):
        logger.info(f"CatPred already exists at {target_path}. Skipping clone.")
    else:
        try:
            logger.info(f"Cloning CatPred into {target_path}...")
            subprocess.run(["git", "clone", repo_url, target_path], check=True)
            logger.info("Clone complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cloning CatPred: {e}")
            raise

    return target_path

def download_catpred_weights(cache_dir):
    target_dir = os.path.join(cache_dir, "CatPred_weights")
    os.makedirs(target_dir, exist_ok=True)

    files_info = {
        "pretrained_production.tar.gz": "https://catpred.s3.us-east-1.amazonaws.com/pretrained_production.tar.gz",
        "processed_databases.tar.gz": "https://catpred.s3.amazonaws.com/processed_databases.tar.gz"
    }

    for filename, url in files_info.items():
        archive_path = os.path.join(target_dir, filename)
        marker_dirname = filename.replace(".tar.gz", "")
        marker_dir = os.path.join(target_dir, 'pretrained')

        kcat_mod = os.path.join(marker_dir, 'production', 'kcat')
        ki_mod = os.path.join(marker_dir, 'production', 'ki')
        km = os.path.join(marker_dir, 'production', 'km')

        if 'pretrained' in filename:
            counter = 0
            for mod_path in [kcat_mod, ki_mod, km]:
                if os.path.exists(mod_path):
                    counter += 1
            if counter == 3:
                continue

        else:
            dbs = ['brenda.csv', 'sabio.csv']
            db_paths = [os.path.join(target_dir, 'processed_databases', db) for db in dbs]
            if all(os.path.exists(db_path) for db_path in db_paths):
                continue

        # Download file if not already present
        if not os.path.exists(archive_path):
            logger.info(f"Downloading {filename}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(archive_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        else:
            logger.info(f"{filename} already downloaded.")

        # Extract archive
        logger.info(f"Extracting {filename}...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=target_dir)

        # Remove archive after extraction
        os.remove(archive_path)
        logger.info(f"Finished processing {filename}.")

    logger.info(f"All required files are present in: {target_dir}")
