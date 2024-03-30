import importlib

import pytorch_lightning as pl
import torch
import argparse
import esm
import time
import gc
import subprocess
import os
from git import Repo
from torch import inf
import sys
import xgboost as xgb
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from Bio import SeqIO
import requests
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
import shutil
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
from rdkit import Chem
# from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
# from fairscale.nn.wrap import enable_wrap, wrap
import builtins
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from trill.utils.lightning_models import ESM, ProtGPT2, CustomWriter, ESM_Gibbs, ProtT5, ZymCTRL, ProstT5, Custom3DiDataset, Ankh
from trill.utils.update_weights import weights_update
from trill.utils.dock_utils import perform_docking, fixer_of_pdbs, write_docking_results_to_file
from trill.utils.simulation_utils import relax_structure, run_simulation
from transformers import AutoTokenizer, EsmForProteinFolding, set_seed
from pytorch_lightning.callbacks import ModelCheckpoint
# from trill.utils.strategy_tuner import tune_esm_inference, tune_esm_train
from trill.utils.protgpt2_utils import ProtGPT2_wrangle
from trill.utils.esm_utils import ESM_IF1_Wrangle, ESM_IF1, convert_outputs_to_pdb, parse_and_save_all_predictions
from trill.utils.visualize import reduce_dims, viz
from trill.utils.MLP import MLP_C2H2, inference_epoch
from sklearn.ensemble import IsolationForest
import skops.io as sio
from sklearn.preprocessing import LabelEncoder
import trill.utils.ephod_utils as eu
from trill.utils.classify_utils import generate_class_key_csv, prep_data, log_results, xg_test, sweep, train_model, custom_xg_test
from trill.utils.fetch_embs import convert_embeddings_to_csv, download_embeddings
from esm.inverse_folding.util import load_coords
import logging
from pyfiglet import Figlet
import bokeh
from Bio import PDB
from icecream import ic
import pkg_resources

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):

    # torch.set_float32_matmul_precision('medium')
    start = time.time()
    f = Figlet(font="graffiti")
    print(f.renderText("TRILL"))
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "name",
        help = "Name of run",
        action = "store"
        )

    
    parser.add_argument(
        "GPUs",
        help="Input total number of GPUs per node",
        action="store",
        default = 1
)

    parser.add_argument(
        "--nodes",
        help="Input total number of nodes. Default is 1",
        action="store",
        default = 1
)
    

    parser.add_argument(
        "--logger",
        help="Enable Tensorboard logger. Default is None",
        action="store",
        default = False,
        dest="logger",
)

    parser.add_argument(
        "--profiler",
        help="Utilize PyTorchProfiler",
        action="store_true",
        default=False,
        dest="profiler",
)
    parser.add_argument(
        "--RNG_seed",
        help="Input RNG seed. Default is 123",
        action="store",
        default = 123
)
    parser.add_argument(
        "--outdir",
        help="Input full path to directory where you want the output from TRILL",
        action="store",
        default = '.'
)

    parser.add_argument(
        "--n_workers",
        help="Change number of CPU cores/'workers' TRILL uses",
        action="store",
        default = 1
)


##############################################################################################################

    subparsers = parser.add_subparsers(dest='command')

    commands = {}
    for command in [
        "embed",
        "finetune",
        "inv_fold_gen",
        "lang_gen",
        "diff_gen",
        "classify",
        "fold",
        "visualize",
        "simulate",
        "dock",
    ]:
        commands[command] = importlib.import_module(f"trill.commands.{command}")
        commands[command].setup(subparsers)


##############################################################################################################

    utils = subparsers.add_parser('utils', help='Misc utilities')

    utils.add_argument(
        "tool",
        help="prepare_class_key: Pepare a csv for use with the classify command. Takes a directory or text file with list of paths for fasta files. Each file will be a unique class, so if your directory contains 5 fasta files, there will be 5 classes in the output key csv.",
        choices = ['prepare_class_key', 'fetch_embeddings']
)

    utils.add_argument(
        "--dir",
        help="Directory to be used for creating a class key csv for classification.",
        action="store",
)

    utils.add_argument(
        "--fasta_paths_txt",
        help="Text file with absolute paths of fasta files to be used for creating the class key. Each unique path will be treated as a unique class, and all the sequences in that file will be in the same class.",
        action="store",
)
    utils.add_argument(
    "--uniprotDB",
    help="UniProt embedding dataset to download.",
    choices=['UniProtKB',
        'A.thaliana',
        'C.elegans',
        'E.coli',
        'H.sapiens',
        'M.musculus',
        'R.norvegicus',
        'SARS-CoV-2'],
    action="store",
)   
    utils.add_argument(
    "--rep",
    help="The representation to download.",
    choices=['per_AA', 'avg'],
    action="store"
)

    
##############################################################################################################

    

    args = parser.parse_args()

    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".trill_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    pl.seed_everything(int(args.RNG_seed))
    set_seed(int(args.RNG_seed))

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    
    
    torch.backends.cuda.matmul.allow_tf32 = True
    if int(args.GPUs) == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if int(args.nodes) <= 0:
            raise Exception(f'There needs to be at least one cpu node to use TRILL')
    #if args.tune == True:
        #data = esm.data.FastaBatchedDataset.from_file(args.query)
        # tune_esm_inference(data)
        # tune_esm_train(data, int(args.GPUs))
    
    else:    
        if args.logger == True:
            logger = TensorBoardLogger("logs")
        else:
            logger = False
        if args.profiler:
            profiler = PyTorchProfiler(filename='test-logs')
        else:
            profiler = None

    if args.command in commands:
        commands[args.command].run(args, logger, profiler)
    else:
        if args.command == 'utils':
            if args.tool == 'prepare_class_key':
                generate_class_key_csv(args)
            elif args.tool == 'fetch_embeddings':
                h5_path = download_embeddings(args)
                h5_name = os.path.splitext(os.path.basename(h5_path))[0]
                convert_embeddings_to_csv(h5_path, os.path.join(args.outdir, f'{h5_name}.csv'))


        



    
    end = time.time()
    print("Finished!")
    print(f"Time elapsed: {end-start} seconds")
 

def cli(args=None):
    if not args:
        args = sys.argv[1:]    
    main(args)
if __name__ == '__main__':
    print("this shouldn't show up...")
