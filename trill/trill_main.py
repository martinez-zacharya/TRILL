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

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

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
    ]:
        commands[command] = importlib.import_module(f"trill.commands.{command}")
        commands[command].setup(subparsers)


##############################################################################################################


    diffuse_gen = subparsers.add_parser('diff_gen', help='Generate proteins using RFDiffusion')

    diffuse_gen.add_argument("--contigs",
        help="Generate proteins between these sizes in AAs for RFDiffusion. For example, --contig 100-200, will result in proteins in this range",
        action="store",
        )

    diffuse_gen.add_argument("--RFDiffusion_Override",
        help="Change RFDiffusion model. For example, --RFDiffusion_Override ActiveSite will use ActiveSite_ckpt.pt for holding small motifs in place. ",
        action="store",
        default = False
        )

    diffuse_gen.add_argument(
        "--num_return_sequences",
        help="Number of sequences for RFDiffusion to generate. Default is 5",
        default=5,
        type=int,
)

    diffuse_gen.add_argument("--Inpaint",
        help="Residues to inpaint.",
        action="store",
        default = None
        )

    diffuse_gen.add_argument("--query",
        help="Input pdb file for motif scaffolding, partial diffusion etc.",
        action="store",
        )

    # diffuse_gen.add_argument("--sym",
    #     help="Use this flag to generate symmetrical oligomers.",
    #     action="store_true",
    #     default=False
    #     )

    # diffuse_gen.add_argument("--sym_type",
    #     help="Define resiudes that binder must interact with. For example, --hotspots A30,A33,A34 , where A is the chain and the numbers are the residue indices.",
    #     action="store",
    #     default=None
    #     )

    diffuse_gen.add_argument("--partial_T",
        help="Adjust partial diffusion sampling value.",
        action="store",
        default=None,
        type=int
        )

    diffuse_gen.add_argument("--partial_diff_fix",
        help="Pass the residues that you want to keep fixed for your input pdb during partial diffusion. Note that the residues should be 0-indexed.",
        action="store",
        default=None
        )

    diffuse_gen.add_argument("--hotspots",
        help="Define resiudes that binder must interact with. For example, --hotspots A30,A33,A34 , where A is the chain and the numbers are the residue indices.",
        action="store",
        default=None
        )


    # diffuse_gen.add_argument("--RFDiffusion_yaml",
    #     help="Specify RFDiffusion params using a yaml file. Easiest option for complicated runs",
    #     action="store",
    #     default = None
    #     )

##############################################################################################################
    classify = subparsers.add_parser('classify', help='Classify proteins using either pretrained classifiers or train/test your own.')

    classify.add_argument(
        "classifier",
        help="Predict thermostability/optimal enzymatic pH using TemStaPro/EpHod or choose custom to train/use your own XGBoost or Isolation Forest classifier. Note for training XGBoost, you need to submit roughly equal amounts of each class as part of your query.",
        choices = ['TemStaPro', 'EpHod', 'XGBoost', 'iForest']
)
    classify.add_argument(
        "query",
        help="Fasta file of sequences to score",
        action="store"
)
    classify.add_argument(
        "--key",
        help="Input a CSV, with your class mappings for your embeddings where the first column is the label and the second column is the class.",
        action="store"
)
    classify.add_argument(
        "--save_emb",
        help="Save csv of ProtT5 embeddings",
        action="store_true",
        default=False
)
    classify.add_argument(
        "--emb_model",
        help="Select desired protein language model for embedding your query proteins to then train your custom classifier. Default is esm2_t12_35M",
        default = 'esm2_t12_35M',
        action="store",
        choices = ['esm2_t6_8M', 'esm2_t12_35M', 'esm2_t30_150M', 'esm2_t33_650M', 'esm2_t36_3B','esm2_t48_15B', 'ProtT5-XL', 'ProstT5', 'Ankh', 'Ankh-Large']
)
    classify.add_argument(
        "--train_split",
        help="Choose your train-test percentage split for training and evaluating your custom classifier. For example, --train .6 would split your input sequences into two groups, one with 60%% of the sequences to train and the other with 40%% for evaluating",
        action="store",
)
    classify.add_argument(
        "--preTrained",
        help="Enter the path to your pre-trained XGBoost binary classifier that you've trained with TRILL. This will be a .json file.",
        action="store",
)

    classify.add_argument(
        "--preComputed_Embs",
        help="Enter the path to your pre-computed embeddings. Make sure they match the --emb_model you select.",
        action="store",
        default=False
)

    classify.add_argument(
        "--batch_size",
        help="EpHod: Sets batch_size for embedding with ESM1v.",
        action="store",
        default=1
)

    classify.add_argument(
        "--xg_gamma",
        help="XGBoost: sets gamma for XGBoost, which is a hyperparameter that sets 'Minimum loss reduction required to make a further partition on a leaf node of the tree.'",
        action="store",
        default=0.4
)

    classify.add_argument(
        "--xg_lr",
        help="XGBoost: Sets the learning rate for XGBoost",
        action="store",
        default=0.2
)

    classify.add_argument(
        "--xg_max_depth",
        help="XGBoost: Sets the maximum tree depth",
        action="store",
        default=8
)


    classify.add_argument(
        "--xg_reg_alpha",
        help="XGBoost: L1 regularization term on weights",
        action="store",
        default=0.8
)

    classify.add_argument(
        "--xg_reg_lambda",
        help="XGBoost: L2 regularization term on weights",
        action="store",
        default=0.1
)
    classify.add_argument(
        "--if_contamination",
        help="iForest: The amount of outliers in the data. Default is automatically determined, but you can set it between (0 , 0.5])",
        action="store",
        default='auto'
)
    classify.add_argument(
        "--n_estimators",
        help="XGBoost/iForest: Number of boosting rounds",
        action="store",
        default=115
)
    classify.add_argument(
        "--sweep",
        help="XGBoost: Use this flag to perform cross-validated bayesian optimization over the hyperparameter space.",
        action="store_true",
        default=False
)
    classify.add_argument(
        "--sweep_cv",
        help="XGBoost: Change the number of folds used for cross-validation.",
        action="store",
        default=3
)
    classify.add_argument(
        "--f1_avg_method",
        help="XGBoost: Change the scoring method used for calculated F1. Default is with no averaging.",
        action="store",
        default=None,
        choices=["macro", "weighted", "micro", "None"]
)
##############################################################################################################
    
    fold = subparsers.add_parser('fold', help='Predict 3D protein structures using ESMFold or obtain 3Di structure for use with Foldseek to perform remote homology detection')

    fold.add_argument("model", 
        help="Choose your desired model.", 
        choices = ['ESMFold', 'ProstT5']
        )
    
    fold.add_argument("query", 
        help="Input fasta file", 
        action="store"
        )
    fold.add_argument("--strategy", 
        help="ESMFold: Choose a specific strategy if you are running out of CUDA memory. You can also pass either 64, or 32 for model.trunk.set_chunk_size(x)", 
        action="store",
        default = None,
        )    

    fold.add_argument(
        "--batch_size",
        help="ESMFold: Change batch-size number for folding proteins. Default is 1",
        action="store",
        default = 1,
        dest="batch_size",
)
##############################################################################################################
    visualize = subparsers.add_parser('visualize', help='Reduce dimensionality of embeddings to 2D')

    visualize.add_argument("embeddings", 
        help="Embeddings to be visualized", 
        action="store"
        )
    
    visualize.add_argument("--method", 
        help="Method for reducing dimensions of embeddings. Default is PCA", 
        action="store",
        choices = ['PCA', 'UMAP', 'tSNE'],
        default="PCA"
        )
    visualize.add_argument("--key", 
        help="Input a CSV, with your group mappings for your embeddings where the first column is the label and the second column is the group to be colored.", 
        action="store",
        default=False
        )
    
##############################################################################################################
    simulate = subparsers.add_parser('simulate', help='Use MD to relax protein structures')

    simulate.add_argument(
        "receptor",
        help="Receptor of interest to be simulated. Must be either pdb file or a .txt file with the absolute path for each pdb, separated by a new-line.",
        action="store",
)

    simulate.add_argument("--ligand", 
        help="Ligand of interest to be simulated with input receptor", 
        action="store",
        )
    
    simulate.add_argument(
        "--constraints",
        help="Specifies which bonds and angles should be implemented with constraints. Allowed values are None, HBonds, AllBonds, or HAngles.",
        choices=["None", "HBonds", "AllBonds", "HAngles"],
        default="None",
        action="store",
    )

    simulate.add_argument(
        "--rigidWater",
        help="If true, water molecules will be fully rigid regardless of the value passed for the constraints argument.",
        default=None,
        action="store_true",
    )

    simulate.add_argument(
        '--forcefield', 
        type=str, 
        default='amber14-all.xml', 
        help='Force field to use. Default is amber14-all.xml'
    )
    
    simulate.add_argument(
        '--solvent', 
        type=str, 
        default='amber14/tip3pfb.xml', 
        help='Solvent model to use, the default is amber14/tip3pfb.xml'
    )
    simulate.add_argument(
        '--solvate', 
        default=False, 
        help='Add to solvate your simulation',
        action='store_true'
    )

    simulate.add_argument(
        '--step_size',
        help='Step size in femtoseconds. Default is 2',
        type=float,
        default=2, 
        action="store",
    )
    simulate.add_argument(
        '--num_steps',
        type=int,
        default=5000,
        help='Number of simulation steps'
    )

    simulate.add_argument(
        '--reporting_interval',
        type=int,
        default=1000,
        help='Reporting interval for simulation'
    )

    simulate.add_argument(
        '--output_traj_dcd',
        type=str,
        default='trajectory.dcd',
        help='Output trajectory DCD file'
    )

    simulate.add_argument(
        '--apply-harmonic-force',
        help='Whether to apply a harmonic force to pull the molecule.',
        type=bool,
        default=False,
        action="store",
    )

    simulate.add_argument(
        '--force-constant',
        help='Force constant for the harmonic force in kJ/mol/nm^2.',
        type=float,
        default=None,
        action="store",
    )

    simulate.add_argument(
        '--z0',
        help='The z-coordinate to pull towards in nm.',
        type=float,
        default=None,
        action="store",
    )

    simulate.add_argument(
        '--molecule-atom-indices',
        help='Comma-separated list of atom indices to which the harmonic force will be applied.',
        type=str,
        default="0,1,2",  # Replace with your default indices
        action="store",
    )

    simulate.add_argument(
        '--equilibration_steps',
        help='Steps you want to take for NVT and NPT equilibration. Each step is 0.002 picoseconds',
        type=int,
        default=300, 
        action="store",
    )

    simulate.add_argument(
        '--periodic_box',
        help='Give, in nm, one of the dimensions to build the periodic boundary.',
        type=int,
        default=10, 
        action="store",
    )
#     simulate.add_argument(
#         '--martini_top',
#         help='Specify the path to the MARTINI topology file you want to use.',
#         type=str,
#         default=False,
#         action="store",
# )
    simulate.add_argument(
        '--just_relax',
        help='Just relaxes the input structure(s) and outputs the fixed and relaxed structure(s). The forcefield that is used is amber14.',
        action="store_true",
        default=False,
    )

    simulate.add_argument(
        '--reporter_interval',
        help='Set interval to save PDB and energy snapshot. Note that the higher the number, the bigger the output files will be and the slower the simulation. Default is 1000',
        action="store",
        default=1000,
    )

##############################################################################################################
    dock = subparsers.add_parser('dock', help='Perform molecular docking with proteins and ligands. Note that you should relax your protein receptor with Simulate or another method before docking.')

    dock.add_argument("algorithm",
        help="Note that while LightDock can dock protein ligands, DiffDock, Smina, and Vina can only do small-molecules.",
        choices = ['DiffDock', 'Vina', 'Smina', 'LightDock', 'GeoDock']
    )

    dock.add_argument("protein", 
        help="Protein of interest to be docked with ligand", 
        action="store"
        )
    
    dock.add_argument("ligand", 
        help="Ligand to dock protein with. Note that with Autodock Vina, you can dock multiple ligands at one time. Simply provide them one after another before any other optional TRILL arguments are added. Also, if a .txt file is provided with each line providing the absolute path to different ligands, TRILL will dock each ligand one at a time.", 
        action="store",
        nargs='*'
        )
    
    # dock.add_argument("--force_ligand", 
    #     help="If you are not doing blind docking, TRILL will automatically assume your ligand is a small molecule if the MW is less than 800. To get around this, you can force TRILL to read the ligand as either type.", 
    #     default=False,
    #     choices = ['small', 'protein']
    #     )
    
    dock.add_argument("--save_visualisation", 
        help="DiffDock: Save a pdb file with all of the steps of the reverse diffusion.", 
        action="store_true",
        default=False
        )
    
    dock.add_argument("--samples_per_complex", 
        help="DiffDock: Number of samples to generate.", 
        type = int,
        action="store",
        default=10
        )
    
    dock.add_argument("--no_final_step_noise", 
        help="DiffDock: Use no noise in the final step of the reverse diffusion", 
        action="store_true",
        default=False
        )
    
    dock.add_argument("--inference_steps", 
        help="DiffDock: Number of denoising steps", 
        type=int,
        action="store",
        default=20
        )

    dock.add_argument("--actual_steps", 
        help="DiffDock: Number of denoising steps that are actually performed", 
        type=int,
        action="store",
        default=None
        )
    dock.add_argument("--min_radius", 
        help="Smina/Vina + Fpocket: Minimum radius of alpha spheres in a pocket. Default is 3Å.", 
        type=float,
        action="store",
        default=3.0
        )

    dock.add_argument("--max_radius", 
        help="Smina/Vina + Fpocket: Maximum radius of alpha spheres in a pocket. Default is 6Å.", 
        type=float,
        action="store",
        default=6.0
        )

    dock.add_argument("--min_alpha_spheres", 
        help="Smina/Vina + Fpocket: Minimum number of alpha spheres a pocket must contain to be considered. Default is 35.", 
        type=int,
        action="store",
        default=35
        )
    
    dock.add_argument("--exhaustiveness", 
        help="Smina/Vina: Change computational effort.", 
        type=int,
        action="store",
        default=8
        )
    
    dock.add_argument("--blind", 
        help="Smina/Vina: Perform blind docking and skip binding pocket prediction with fpocket", 
        action="store_true",
        default=False
        )
    dock.add_argument("--anm", 
        help="LightDock: If selected, backbone flexibility is modeled using Anisotropic Network Model (via ProDy)", 
        action="store_true",
        default=False
        )
    
    dock.add_argument("--swarms", 
        help="LightDock: The number of swarms of the simulations, default is 25", 
        action="store",
        type=int,
        default=25
        )
    
    dock.add_argument("--sim_steps", 
        help="LightDock: The number of steps of the simulation. Default is 100", 
        action="store",
        type=int,
        default=100
        )
    dock.add_argument("--restraints", 
        help="LightDock: If restraints_file is provided, residue restraints will be considered during the setup and the simulation", 
        action="store",
        default=None
        )
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
    def process_sublist(sublist):
        if isinstance(sublist, tuple) and len(sublist) == 2:
            return [sublist]
        elif isinstance(sublist, list):
            return sublist
        else:
            print(f"Unexpected data structure: {sublist=}")
        return []

    if args.command in commands:
        commands[args.command].run(args, logger, profiler)
    else:


        if args.command == 'visualize':
            reduced_df, incsv = reduce_dims(args.name, args.embeddings, args.method)
            layout = viz(reduced_df, args)
            bokeh.io.output_file(filename=os.path.join(args.outdir, f'{args.name}_{args.method}_{incsv}.html'), title=args.name)
            bokeh.io.save(layout, filename=os.path.join(args.outdir, f'{args.name}_{args.method}_{incsv}.html'), title = args.name)


        elif args.command == 'diff_gen':
            # command = "conda install -c dglteam dgl-cuda11.7 -y -S -q".split(' ')
            # subprocess.run(command, check = True)
            print('Finding RFDiffusion weights... \n')
            if not os.path.exists((os.path.join(cache_dir, 'RFDiffusion_weights'))):
                os.makedirs(os.path.join(cache_dir, 'RFDiffusion_weights'))

                commands = [
                'wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt',
                'wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt',
                'wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt',
                'wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt',
                'wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt',
                'wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt',
                'wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt'
                    ]
                for command in commands:
                    if not os.path.isfile(os.path.join(cache_dir, f'RFDiffusion_weights/{command.split("/")[-1]}')):
                        subprocess.run(command.split(' '))
                        subprocess.run(['mv', command.split("/")[-1], os.path.join(cache_dir, 'RFDiffusion_weights')])

            if not os.path.exists(os.path.join(cache_dir, 'RFDiffusion')):
                print('Cloning forked RFDiffusion')
                os.makedirs(os.path.join(cache_dir, 'RFDiffusion'))
                rfdiff = Repo.clone_from('https://github.com/martinez-zacharya/RFDiffusion', os.path.join(cache_dir, 'RFDiffusion/'))
                rfdiff_git_root = rfdiff.git.rev_parse("--show-toplevel")
                subprocess.run(['pip', 'install', '-e', rfdiff_git_root])
                command = f'pip install {rfdiff_git_root}/env/SE3Transformer'.split(' ')
                subprocess.run(command)
                sys.path.insert(0, os.path.join(cache_dir, 'RFDiffusion'))

            else:
                sys.path.insert(0, os.path.join(cache_dir, 'RFDiffusion'))
                git_repo = Repo(os.path.join(cache_dir, 'RFDiffusion'), search_parent_directories=True)
                rfdiff_git_root = git_repo.git.rev_parse("--show-toplevel")

            from run_inference import run_rfdiff
            # if args.sym:
            #     run_rfdiff((f'{rfdiff_git_root}/config/inference/symmetry.yaml'), args)
            # else:
            #     run_rfdiff((f'{rfdiff_git_root}/config/inference/base.yaml'), args)
            run_rfdiff((f'{rfdiff_git_root}/config/inference/base.yaml'), args)

        elif args.command == 'fold':
            if args.model == 'ESMFold':
                data = esm.data.FastaBatchedDataset.from_file(args.query)
                tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
                if int(args.GPUs) == 0:
                    model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1', low_cpu_mem_usage=True, torch_dtype='auto')
                else:
                    model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1', device_map='sequential', torch_dtype='auto')
                    model = model.cuda()
                    model.esm = model.esm.half()
                    model = model.cuda()
                if args.strategy != None:
                    model.trunk.set_chunk_size(int(args.strategy))
                fold_df = pd.DataFrame(list(data), columns = ["Entry", "Sequence"])
                sequences = fold_df.Sequence.tolist()
                with torch.no_grad():
                    for input_ids in tqdm(range(0, len(sequences), int(args.batch_size))):
                        i = input_ids
                        batch_input_ids = sequences[i: i + int(args.batch_size)]
                        if int(args.GPUs) == 0:
                            if int(args.batch_size) > 1:
                                tokenized_input = tokenizer(batch_input_ids, return_tensors="pt", add_special_tokens=False, padding=True)['input_ids']
                            else:
                                tokenized_input = tokenizer(batch_input_ids, return_tensors="pt", add_special_tokens=False)['input_ids']
                            tokenized_input = tokenized_input.clone().detach()
                            prot_len = len(batch_input_ids[0])
                            try:
                                output = model(tokenized_input)
                                output = {key: val.cpu() for key, val in output.items()}
                            except RuntimeError as e:
                                    if 'out of memory' in str(e):
                                        print(f'Protein too long to fold for current hardware: {prot_len} amino acids long)')
                                        print(e)
                                    else:
                                        print(e)
                                        pass
                        else:
                            if int(args.batch_size) > 1:
                                tokenized_input = tokenizer(batch_input_ids, return_tensors="pt", add_special_tokens=False, padding=True)['input_ids']
                                prot_len = len(batch_input_ids[0])
                            else:
                                tokenized_input = tokenizer(batch_input_ids, return_tensors="pt", add_special_tokens=False)['input_ids']
                                prot_len = len(batch_input_ids[0])
                            tokenized_input = tokenized_input.clone().detach()
                            try:
                                tokenized_input = tokenized_input.to(model.device)
                                output = model(tokenized_input)
                                output = {key: val.cpu() for key, val in output.items()}
                            except RuntimeError as e:
                                    if 'out of memory' in str(e):
                                        print(f'Protein too long to fold for current hardware: {prot_len} amino acids long)')
                                        print(e)
                                    else:
                                        print(e)
                                        pass
                        output = convert_outputs_to_pdb(output)
                        if int(args.batch_size) > 1:
                            start_idx = i
                            end_idx = i + int(args.batch_size)
                            identifier = fold_df.Entry[start_idx:end_idx].tolist()
                        else:
                            identifier = [fold_df.Entry[i]]
                        for out, iden in zip(output, identifier):
                            with open(os.path.join(args.outdir,f"{iden}.pdb"), "w") as f:
                                f.write("".join(out))

            elif args.model == "ProstT5":
                model = ProstT5(args)
                data = esm.data.FastaBatchedDataset.from_file(args.query)
                dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = int(args.batch_size), num_workers=0)
                pred_writer = CustomWriter(output_dir=args.outdir, write_interval="epoch")
                if int(args.GPUs) == 0:
                    trainer = pl.Trainer(enable_checkpointing=False, callbacks = [pred_writer], logger=logger, num_nodes=int(args.nodes))
                else:
                    trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), callbacks = [pred_writer], accelerator='gpu', logger=logger, num_nodes=int(args.nodes))

                reps = trainer.predict(model, dataloader)
                cwd_files = os.listdir(args.outdir)
                pt_files = [file for file in cwd_files if 'predictions_' in file]
                pred_embeddings = []
                if args.batch_size == 1 or int(args.GPUs) > 1:
                    for pt in pt_files:
                        preds = torch.load(os.path.join(args.outdir, pt))
                        for pred in preds:
                            for sublist in pred:
                                if len(sublist) == 2 and args.batch_size == 1:
                                    pred_embeddings.append(tuple([sublist[0], sublist[1]]))
                                else:
                                    processed_sublists = process_sublist(sublist)
                                    for sub in processed_sublists:

                                        pred_embeddings.append(tuple([sub[0], sub[1]]))
                    embedding_df = pd.DataFrame(pred_embeddings, columns = ['3Di', 'Label'])
                    finaldf = embedding_df['3Di'].apply(pd.Series)
                    finaldf['Label'] = embedding_df['Label']
                else:
                    embs = []
                    for rep in reps:
                        inner_embeddings = [item[0] for item in rep]
                        inner_labels = [item[1] for item in rep]
                        for emb_lab in zip(inner_embeddings, inner_labels):
                            embs.append(emb_lab)
                    embedding_df = pd.DataFrame(embs, columns = ['3Di', 'Label'])
                    finaldf = embedding_df['3Di'].apply(pd.Series)
                    finaldf['Label'] = embedding_df['Label']


                outname = os.path.join(args.outdir, f'{args.name}_{args.model}.csv')
                finaldf.to_csv(outname, index = False, header=['3Di', 'Label'])
                for file in pt_files:
                    os.remove(os.path.join(args.outdir,file))

        elif args.command == 'dock':
            ligands = []
            if isinstance(args.ligand, list) and len(args.ligand) > 1:
                for lig in args.ligand:
                    ligands.append(lig)
                    args.multi_lig = True
            else:
                args.ligand = args.ligand[0]
                args.multi_lig = False
                if args.ligand.endswith('.txt'):
                    with open(args.ligand, 'r') as infile:
                        for path in infile:
                            path = path.strip()
                            if not path:
                                continue
                            ligands.append(path)
                else:
                    ligands.append(args.ligand)

            protein_name = os.path.splitext(os.path.basename(args.protein))[0]


            if args.algorithm == 'Smina' or args.algorithm == 'Vina':
                docking_results = perform_docking(args, ligands)
                write_docking_results_to_file(docking_results, args, protein_name, args.algorithm)
            elif args.algorithm == 'LightDock':
                perform_docking(args, ligands)
                print(f"LightDock run complete! Output files are in {args.outdir}")
            elif args.algorithm == 'GeoDock':
                try:
                    pkg_resources.get_distribution('geodock')
                except pkg_resources.DistributionNotFound:
                    install_cmd = 'pip install git+https://github.com/martinez-zacharya/GeoDock.git'.split(' ')
                    subprocess.run(install_cmd)
                from geodock.GeoDockRunner import GeoDockRunner, EnMasseGeoDockRunner
                base_url = "https://raw.githubusercontent.com/martinez-zacharya/GeoDock/main/geodock/weights/dips_0.3.ckpt"
                weights_path = f'{cache_dir}/dips_0.3.ckpt'
                if not os.path.exists(weights_path):
                    r = requests.get(base_url)
                    with open(weights_path, "wb") as file:
                        file.write(r.content)

                rec_coord, rec_seq = load_coords(args.protein, chain=None)
                rec_name = os.path.basename(args.protein).split('.')[0]

                lig_seqs = []
                lig_coords = []
                lig_names = []
                with open(f'tmp_master.fasta', 'w+') as fasta:
                    fasta.write(f'>{rec_name}\n')
                    fasta.write(f'{rec_seq}\n')
                    for lig in ligands:
                        lig_name = os.path.basename(lig).split('.')[0]
                        coords, seq = load_coords(lig, chain=None)
                        coords = torch.nan_to_num(torch.from_numpy(coords))
                        lig_seqs.append(seq)
                        lig_coords.append(coords)
                        lig_names.append(lig_name)
                        fasta.write(f'>{lig_name}\n')
                        fasta.write(f'{seq}\n')

                model_import_name = f'esm.pretrained.esm2_t33_650M_UR50D()'
                args.per_AA = True
                args.avg = False
                model = ESM(eval(model_import_name), 0.0001, args)
                seq_data = esm.data.FastaBatchedDataset.from_file('tmp_master.fasta')
                loader = torch.utils.data.DataLoader(seq_data, shuffle = False, batch_size = 1, num_workers=0, collate_fn=model.alphabet.get_batch_converter())
                pred_writer = CustomWriter(output_dir=args.outdir, write_interval="epoch")
                if int(args.GPUs) == 0:
                    trainer = pl.Trainer(enable_checkpointing=False, callbacks = [pred_writer], logger=logger, num_nodes=int(args.nodes))
                else:
                    trainer = pl.Trainer(enable_checkpointing=False, precision=16, devices=int(args.GPUs), callbacks = [pred_writer], accelerator='gpu', logger=logger, num_nodes=int(args.nodes))

                trainer.predict(model, loader)
                parse_and_save_all_predictions(args)
                master_embs = []
                emb_file = torch.load(f'{args.outdir}/predictions_0.pt')
                for entry in emb_file[0]:
                    emb = entry[0][0][0]
                    master_embs.append(emb)

                rec_emb = master_embs.pop(0)
                for lig_name, lig_seq, lig_coord, lig_emb in zip(lig_names, lig_seqs, lig_coords, master_embs):
                    em_geodock = EnMasseGeoDockRunner(args, ckpt_file=weights_path)
                    pred = em_geodock.dock(
                        rec_info = [rec_name, rec_seq, rec_coord, rec_emb],
                        lig_info = [lig_name, lig_seq, lig_coord, lig_emb],
                        out_name = args.name + '_' + rec_name + '_' + lig_name
                    )
                os.remove(f'{args.outdir}/predictions_0.pt')

            elif args.algorithm == 'DiffDock':
                if not os.path.exists(os.path.join(cache_dir, 'DiffDock')):
                    print('Cloning forked DiffDock')
                    os.makedirs(os.path.join(cache_dir, 'DiffDock'))
                    diffdock = Repo.clone_from('https://github.com/martinez-zacharya/DiffDock', os.path.join(cache_dir, 'DiffDock'))
                    diffdock_root = diffdock.git.rev_parse("--show-toplevel")
                    subprocess.run(['pip', 'install', '-e', diffdock_root])
                    sys.path.insert(0, os.path.join(cache_dir, 'DiffDock'))
                else:
                    sys.path.insert(0, os.path.join(cache_dir, 'DiffDock'))
                    diffdock = Repo(os.path.join(cache_dir, 'DiffDock'))
                    diffdock_root = diffdock.git.rev_parse("--show-toplevel")
                from inference import run_diffdock
                run_diffdock(args, diffdock_root)

                    # out_dir = os.path.join(args.outdir, f'{args.name}_DiffDock_out')
                    # rec = args.protein.split('.')[-2]
                    # out_rec = rec.split('/')[-1]
                    # convert_rec = f'obabel {rec}.pdb -O {out_rec}.pdbqt'.split(' ')
                    # subprocess.run(convert_rec, stdout=subprocess.DEVNULL)
                    # for file in os.listdir(out_dir):
                    #     if 'confidence' in file:
                    #         file_pre = file.split('.sdf')[-2]
                    #         convert_lig = f'obabel {out_dir}/{file} -O {file_pre}.pdbqt'.split(' ')
                    #         subprocess.run(convert_lig, stdout=subprocess.DEVNULL)

                    #         smina_cmd = f'smina --score_only -r {out_rec}.pdbqt -l {file_pre}.pdbqt'.split(' ')
                    #         result = subprocess.run(smina_cmd, stdout=subprocess.PIPE)

                    #         result = re.search("Affinity: \w+.\w+", result.stdout.decode('utf-8'))
                    #         affinity = result.group()
                    #         affinity = re.search('\d+\.\d+', affinity).group()


        elif args.command == 'classify':
            if args.sweep and not args.train_split:
                raise Exception("You need to provide a train-test fraction with --train_split!")
            if args.classifier == 'TemStaPro':
                if not args.preComputed_Embs:
                    data = esm.data.FastaBatchedDataset.from_file(args.query)
                    model = ProtT5(args)
                    dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = 1, num_workers=0)
                    if int(args.GPUs) > 0:
                        trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), accelerator='gpu', logger=logger, num_nodes=int(args.nodes))
                    else:
                        trainer = pl.Trainer(enable_checkpointing=False, logger=logger, num_nodes=int(args.nodes))
                    reps = trainer.predict(model, dataloader)
                    parse_and_save_all_predictions(args)
                if not os.path.exists(os.path.join(cache_dir, 'TemStaPro_models')):
                    temstapro_models = Repo.clone_from('https://github.com/martinez-zacharya/TemStaPro_models', os.path.join(cache_dir, 'TemStaPro_models'))
                    temstapro_models_root = temstapro_models.git.rev_parse("--show-toplevel")
                else:
                    temstapro_models = Repo(os.path.join(cache_dir, 'TemStaPro_models'))
                    temstapro_models_root = temstapro_models.git.rev_parse("--show-toplevel")
                THRESHOLDS = ["40", "45", "50", "55", "60", "65"]
                SEEDS = ["41", "42", "43", "44", "45"]
                if not args.preComputed_Embs:
                    emb_df = pd.read_csv(os.path.join(args.outdir, f'{args.name}_ProtT5_AVG.csv'))
                else:
                    emb_df = pd.read_csv(args.preComputed_Embs)
                embs = emb_df[emb_df.columns[:-1]].applymap(lambda x: torch.tensor(x)).values.tolist()
                labels = emb_df.iloc[:,-1]
                list_of_tensors = [torch.tensor(l) for l in embs]
                input_data = list(zip(list_of_tensors, labels))
                custom_dataset = CustomDataset(input_data)
                emb_loader = torch.utils.data.DataLoader(custom_dataset, shuffle=False, batch_size=1, num_workers = 0)
                inferences = {}
                for thresh in THRESHOLDS:
                    threshold_inferences = {}
                    for seed in SEEDS:
                        clf = MLP_C2H2(1024, 512, 256)
                        clf.load_state_dict(torch.load(f'{temstapro_models_root}/mean_major_imbal-{thresh}_s{seed}.pt'))
                        clf.eval()
                        if int(args.GPUs) > 0:
                            clf.to('cuda')
                            threshold_inferences[seed] = inference_epoch(clf, emb_loader, device='cuda')
                        else:
                            threshold_inferences[seed] = inference_epoch(clf, emb_loader, device='cpu')
                    for seq in threshold_inferences["41"].keys():
                        mean_prediction = 0
                        for seed in SEEDS:
                            mean_prediction += threshold_inferences[seed][seq]
                        mean_prediction /= len(SEEDS)
                        binary_pred = builtins.round(mean_prediction)
                        inferences[f'{seq}$%#{thresh}'] = (mean_prediction, binary_pred)
                inference_df = pd.DataFrame.from_dict(inferences, orient='index', columns=['Mean_Pred', 'Binary_Pred'])
                inference_df = inference_df.reset_index(names='RawLab')
                inference_df['Protein'] = inference_df['RawLab'].apply(lambda x: x.split('$%#')[0])
                inference_df['Threshold'] = inference_df['RawLab'].apply(lambda x: x.split('$%#')[-1])
                inference_df = inference_df.drop(columns='RawLab')
                inference_df = inference_df[['Protein', 'Threshold', 'Mean_Pred', 'Binary_Pred']]
                inference_df.to_csv(os.path.join(args.outdir, f'{args.name}_TemStaPro_preds.csv'), index = False)
                if not args.save_emb:
                    os.remove(os.path.join(args.outdir, f'{args.name}_ProtT5_AVG.csv'))

            elif args.classifier == 'EpHod':
                logging.getLogger("pytorch_lightning.utilities.rank_zero").addHandler(logging.NullHandler())
                logging.getLogger("pytorch_lightning.accelerators.cuda").addHandler(logging.NullHandler())
                if not os.path.exists(os.path.join(cache_dir, 'EpHod_Models')):
                    cmd = ['curl', '-o', 'saved_models.tar.gz', '--progress-bar', 'https://zenodo.org/record/8011249/files/saved_models.tar.gz?download=1']
                    result = subprocess.run(cmd)

                    cmd = f'mv saved_models.tar.gz {cache_dir}/'.split()

                    subprocess.run(cmd)
                    tarfile = os.path.join(cache_dir, 'saved_models.tar.gz')
                    _ = subprocess.call(f"tar -xvzf {tarfile}", shell=True)
                    _ = subprocess.call(f"rm -rfv {tarfile}", shell=True)
                else:
                    headers, sequences = eu.read_fasta(args.query)
                    accessions = [head.split()[0] for head in headers]
                    headers, sequences, accessions = [np.array(item) for item in \
                                                    (headers, sequences, accessions)]
                    assert len(accessions) == len(headers) == len(sequences), 'Fasta file has unequal headers and sequences'
                    numseqs = len(sequences)


                    # Check sequence lengths
                    lengths = np.array([len(seq) for seq in sequences])
                    long_count = np.sum(lengths > 1022)
                    warning = f"{long_count} sequences are longer than 1022 residues and will be omitted"

                    # Omit sequences longer than 1022
                    if max(lengths) > 1022:
                        print(warning)
                        locs = np.argwhere(lengths <= 1022).flatten()
                        headers, sequences, accessions = [array[locs] for array in \
                                                        (headers, sequences, accessions)]
                        numseqs = len(sequences)

                if not os.path.exists(args.outdir):
                    os.makedirs(args.outdir)

                # Prediction output file
                phout_file = f'{args.outdir}/{args.name}_EpHod.csv'
                embed_file = f'{args.outdir}/{args.name}_ESM1v_embeddings.csv'
                ephod_model = eu.EpHodModel(args)
                num_batches = int(np.ceil(numseqs / args.batch_size))
                all_ypred, all_emb_ephod = [], []
                batches = range(1, num_batches + 1)
                batches = tqdm(batches, desc="Predicting pHopt")
                for batch_step in batches:
                    start_idx = (batch_step - 1) * args.batch_size
                    stop_idx = batch_step * args.batch_size
                    accs = accessions[start_idx : stop_idx]
                    seqs = sequences[start_idx : stop_idx]

                    # Predict with EpHod model
                    ypred, emb_ephod, attention_weights = ephod_model.batch_predict(accs, seqs, args)
                    all_ypred.extend(ypred.to('cpu').detach().numpy())
                    all_emb_ephod.extend(emb_ephod.to('cpu').detach().numpy())

                if args.save_emb:
                    all_emb_ephod = pd.DataFrame(np.array(all_emb_ephod), index=accessions)
                    all_emb_ephod.to_csv(embed_file)

                all_ypred = pd.DataFrame(all_ypred, index=accessions, columns=['pHopt'])
                all_ypred = all_ypred.reset_index(drop=False)
                all_ypred.rename(columns={'index': 'Label'}, inplace=True)
                all_ypred.to_csv(phout_file, index=False)


            elif args.classifier == 'XGBoost':
                outfile = os.path.join(args.outdir, f'{args.name}_XGBoost.out')
                if not args.preComputed_Embs:
                    embed_command = f"trill {args.name} {args.GPUs} --outdir {args.outdir} embed {args.emb_model} {args.query} --avg"
                    subprocess.run(embed_command.split(' '), check=True)
                    df = pd.read_csv(os.path.join(args.outdir, f'{args.name}_{args.emb_model}_AVG.csv'))
                else:
                    df = pd.read_csv(args.preComputed_Embs)

                if args.train_split is not None:
                    le = LabelEncoder()
                    # print(df)
                    train_df, test_df, n_classes = prep_data(df, args)
                    unique_c = np.unique(test_df['NewLab'])
                    classes = train_df['NewLab'].unique()
                    train_df['NewLab'] = le.fit_transform(train_df['NewLab'])
                    test_df['NewLab'] = le.transform(test_df['NewLab'])
                    command_line_args = sys.argv
                    command_line_str = ' '.join(command_line_args)

                    if args.sweep:
                        sweeped_clf = sweep(train_df, args)
                        precision, recall, fscore, support = xg_test(sweeped_clf, le, test_df, args)
                        log_results(outfile, command_line_str, n_classes, args, classes = unique_c, sweeped_clf=sweeped_clf, precision=precision, recall=recall, fscore=fscore, support=support)
                    else:
                        clf = train_model(train_df, args)
                        clf.save_model(os.path.join(args.outdir, f'{args.name}_XGBoost_{len(train_df.columns)-2}.json'))
                        precision, recall, fscore, support = xg_test(clf, le, test_df, args)
                        log_results(outfile, command_line_str, n_classes, args, classes = classes, precision=precision, recall=recall, fscore=fscore, support=support)

                    if not args.save_emb and not args.preComputed_Embs:
                        os.remove(os.path.join(args.outdir, f'{args.name}_{args.emb_model}_AVG.csv'))

                else:
                    if not args.preTrained:
                        raise Exception('You need to provide a model with --args.preTrained to perform inference!')
                    else:
                        clf = xgb.XGBClassifier()
                        clf.load_model(args.preTrained)
                        custom_xg_test(clf, df, args)

                        if not args.save_emb and not args.preComputed_Embs:
                            os.remove(os.path.join(args.outdir, f'{args.name}_{args.emb_model}_AVG.csv'))

            elif args.classifier == 'iForest':
                # Load embeddings
                if not args.preComputed_Embs:
                    embed_command = f"trill {args.name} {args.GPUs} --outdir {args.outdir} embed {args.emb_model} {args.query} --avg".split(' ')
                    subprocess.run(embed_command, check=True)
                    df = pd.read_csv(os.path.join(args.outdir, f'{args.name}_{args.emb_model}_AVG.csv'))
                else:
                    df = pd.read_csv(args.preComputed_Embs)

                # Filter fasta file
                if args.preComputed_Embs and not args.preTrained:
                    valid_labels = set(df['Label'])
                    filtered_records_labels = {record.id for record in SeqIO.parse(args.query, "fasta") if record.id in valid_labels}
                    df = df[df['Label'].isin(filtered_records_labels)]

                # Train or load model
                if not args.preTrained:
                    model = IsolationForest(
                        random_state=int(args.RNG_seed),
                        verbose=True,
                        n_estimators=int(args.n_estimators),
                        contamination=float(args.if_contamination) if args.if_contamination != 'auto' else 'auto'
                    )
                    model.fit(df.iloc[:,:-1])
                    sio.dump(model, os.path.join(args.outdir, f'{args.name}_iForest.skops'))
                else:
                    model = sio.load(args.preTrained, trusted=True)

                    # Predict and output results
                    preds = model.predict(df.iloc[:,:-1])
                    # unique_values, counts = np.unique(preds, return_counts=True)
                    # for value, count in zip(unique_values, counts):
                    #     print(f'{value}: {count}')

                    df['Predicted_Class'] = preds
                    out_df = df[['Label', 'Predicted_Class']]
                    out_df.to_csv(os.path.join(args.outdir, f'{args.name}_iForest_predictions.csv'), index=False)
                if not args.save_emb and not args.preComputed_Embs:
                    os.remove(os.path.join(args.outdir, f'{args.name}_{args.emb_model}_AVG.csv'))


        elif args.command == 'utils':
            if args.tool == 'prepare_class_key':
                generate_class_key_csv(args)
            elif args.tool == 'fetch_embeddings':
                h5_path = download_embeddings(args)
                h5_name = os.path.splitext(os.path.basename(h5_path))[0]
                convert_embeddings_to_csv(h5_path, os.path.join(args.outdir, f'{h5_name}.csv'))

        elif args.command == 'simulate':
            if args.just_relax:
                args.forcefield = 'amber14-all.xml'
                args.solvent = 'amber14/tip3pfb.xml'
                pdb_list = []
                if args.receptor.endswith('.txt'):
                    with open(args.receptor, 'r') as infile:
                        for path in infile:
                            path = path.strip()
                            if not path:
                                continue
                            pdb_list.append(path)
                else:
                    pdb_list.append(args.receptor)
                args.receptor = pdb_list
                fixed_pdb_files = fixer_of_pdbs(args)
                relax_structure(args, fixed_pdb_files)
            else:
                # # print('Currently, Simulate only supports relaxing a structure! Stay tuned for more MD related features...')
                # if args.martini_top:
                #     args.output_traj_dcd = os.path.join(args.outdir, args.output_traj_dcd)
                #     run_simulation(args)
                # else:
                fixed_pdb_files = fixer_of_pdbs(args)

                args.output_traj_dcd = os.path.join(args.outdir, args.output_traj_dcd)

                # Run the simulation on the combined PDB file
                args.protein = fixed_pdb_files[0]
                run_simulation(args)



        



    
    end = time.time()
    print("Finished!")
    print(f"Time elapsed: {end-start} seconds")
 

def cli(args=None):
    if not args:
        args = sys.argv[1:]    
    main(args)
if __name__ == '__main__':
    print("this shouldn't show up...")
