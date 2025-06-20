# Utils for working with MMELON
import sys
import os
import subprocess
from loguru import logger
from tqdm import tqdm
from .molt5_utils import make_embedding_dataframe

def clone_and_install_mmelon(cache_dir):
    repo_url = "https://github.com/BiomedSciAI/biomed-multi-view"
    target_path = os.path.join(cache_dir, "biomed-multi-view")

    # Clone MMELON if needed
    if os.path.exists(target_path):
        logger.info(f"MMELON already exists at {target_path}. Skipping clone.")
    else:
        try:
            logger.info(f"Cloning MMELON into {target_path}...")
            subprocess.run(["git", "clone", repo_url, target_path], check=True)
            logger.info("Clone complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cloning MMELON: {e}")
            raise

    sys.path.append(target_path)
    return

def run_mmelon(args, smiles_list, headers_list):
    from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel, PredictionIterator

    averaged_embeddings = []
    token_level_embeddings = []
    poolparti_embeddings = []
    all_attentions = []

    for smiles, header in tqdm(zip(smiles_list, headers_list)):
        emb = SmallMoleculeMultiViewModel.get_embeddings(
            smiles=smiles,
            model_path='ibm/biomed.sm.mv-te-84m',
            huggingface=True,
        )
        averaged_embeddings.append(emb.reshape(1, -1))
    df = make_embedding_dataframe(headers_list, averaged_embeddings)
    return df

