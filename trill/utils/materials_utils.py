# Utils for interfacing with IBM's Materials repo

import sys
import os
import subprocess
from loguru import logger
from tqdm import tqdm
from .molt5_utils import make_embedding_dataframe

def clone_and_install_fm4m(cache_dir):
    repo_url = "https://github.com/martinez-zacharya/materials"
    target_path = os.path.join(cache_dir, "materials")

    # Clone materials if needed
    if os.path.exists(target_path):
        logger.info(f"Materials already exists at {target_path}. Skipping clone.")
    else:
        try:
            logger.info(f"Cloning Materials into {target_path}...")
            subprocess.run(["git", "clone", repo_url, target_path], check=True)
            logger.info("Clone complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cloning Materials: {e}")
            raise

    sys.path.append(target_path)
    return

def run_mat_ted(args, smiles_list, headers_list):
    import models.fm4m as fm4m

    averaged_embeddings = []
    token_level_embeddings = []
    poolparti_embeddings = []
    all_attentions = []

    for smiles, header in tqdm(zip(smiles_list, headers_list)):
        if 'SELFIES' in args.model:
            emb = fm4m.get_representation([smiles], [smiles], model_type=f'{args.model}')[0]
        else:
            emb = fm4m.get_representation(smiles, smiles, model_type=f'{args.model}')[0]
        averaged_embeddings.append(emb.reshape(1, -1))

    df = make_embedding_dataframe(headers_list, averaged_embeddings)
    return df