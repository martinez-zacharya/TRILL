# Utils for CataPro
import os
import subprocess
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import pandas as pd
from Bio import SeqIO
from itertools import product
from loguru import logger

def run_catapro_prediction(args, cache_dir):
    predict_script = os.path.join(cache_dir, "CataPro", "inference", "predict.py")
    inp_fpath = os.path.join(args.outdir, f"{args.name}_CataPro_input.csv")
    model_dpath = os.path.join(cache_dir, "CataPro", "models")
    device = "cuda" if int(args.GPUs) >= 1 else "cpu"
    out_fpath = os.path.join(args.outdir, f"{args.name}_CataPro_predictions.csv")

    cmd = [
        "python3", predict_script,
        "-inp_fpath", inp_fpath,
        "-model_dpath", model_dpath,
        "-batch_size", "1",
        "-device", device,
        "-out_fpath", out_fpath
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
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
                "Enzyme_id": seq_id,
                "type": "wild",
                "sequence": seq,
                "smiles": smiles
            })

    return pd.DataFrame(rows)

def fetch_prott5():
    model_name = "Rostlab/prot_t5_xl_uniref50"

    # Download model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Locate cache path using a known file
    config_file = hf_hub_download(repo_id=model_name, filename="config.json")
    cache_dir = os.path.dirname(config_file)

    return cache_dir

def fetch_molt5():
    model_name = "laituan245/molt5-base-smiles2caption"
    
    # Download model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    config_file = hf_hub_download(repo_id=model_name, filename="config.json")
    
    cache_dir = os.path.dirname(config_file)

    return cache_dir

def clone_and_install_catapro(cache_dir):
    repo_url = "https://github.com/martinez-zacharya/CataPro"
    target_path = os.path.join(cache_dir, "CataPro")

    if os.path.exists(target_path):
        logger.info(f"CataPro already exists at {target_path}. Skipping clone.")
    else:
        try:
            logger.info(f"Cloning CataPro into {target_path}...")
            subprocess.run(["git", "clone", repo_url, target_path], check=True, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            logger.info("Clone complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cloning CataPro: {e}")
            raise

    return target_path
    