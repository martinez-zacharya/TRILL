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
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
import yaml
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from trill.utils.lightning_models import ESM, ProtGPT2, CustomWriter, ESM_Gibbs, ProtT5, ZymCTRL
from trill.utils.update_weights import weights_update
from transformers import AutoTokenizer, EsmForProteinFolding, set_seed
from pytorch_lightning.callbacks import ModelCheckpoint
# from trill.utils.strategy_tuner import tune_esm_inference, tune_esm_train
from trill.utils.protgpt2_utils import ProtGPT2_wrangle
from trill.utils.esm_utils import ESM_IF1_Wrangle, ESM_IF1, convert_outputs_to_pdb
from trill.utils.visualize import reduce_dims, viz
from trill.utils.MLP import MLP_C2H2, inference_epoch

from pyfiglet import Figlet
import bokeh
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


##############################################################################################################

    subparsers = parser.add_subparsers(dest='command')

    embed = subparsers.add_parser('embed', help='Embed proteins of interest')

    embed.add_argument(
        "model",
        help="You can choose from either 'esm2_t6_8M', 'esm2_t12_35M', 'esm2_t30_150M', 'esm2_t33_650M', 'esm2_t36_3B','esm2_t48_15B', or 'ProtT5-XL'",
        action="store",
        # default = 'esm2_t12_35M_UR50D',
)

    embed.add_argument("query", 
        help="Input fasta file", 
        action="store"
)
    embed.add_argument(
        "--batch_size",
        help="Change batch-size number for embedding proteins. Default is 1",
        action="store",
        default = 1,
        dest="batch_size",
)

    embed.add_argument(
        "--finetuned",
        help="Input path to your own finetuned ESM model",
        action="store",
        default = False,
        dest="finetuned",
)
##############################################################################################################

    finetune = subparsers.add_parser('finetune', help='Finetune protein language models')

    finetune.add_argument(
        "model",
        help="You can choose to finetune either 'esm2_t6_8M', 'esm2_t12_35M', 'esm2_t30_150M', 'esm2_t33_650M', 'esm2_t36_3B','esm2_t48_15B', 'ProtGPT2', or ZymCTRL.",
        action="store",
)

    finetune.add_argument("query", 
        help="Input fasta file", 
        action="store"
)
    finetune.add_argument("--epochs", 
        help="Number of epochs for fine-tuning. Default is 20", 
        action="store",
        default=10,
        dest="epochs",
        )
    finetune.add_argument("--save_on_epoch", 
        help="Saves a checkpoint on every successful epoch completed. WARNING, this could lead to rapid storage consumption", 
        action="store_true",
        default=False,
        )
    finetune.add_argument(
        "--lr",
        help="Learning rate for optimizer. Default is 0.0001",
        action="store",
        default=0.0001,
        dest="lr",
)

    finetune.add_argument(
        "--batch_size",
        help="Change batch-size number for fine-tuning. Default is 1",
        action="store",
        default = 1,
        dest="batch_size",
)

    finetune.add_argument(
        "--strategy",
        help="Change training strategy. Default is None. List of strategies can be found at https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html",
        action="store",
        default = None,
        dest="strategy",
)

    finetune.add_argument(
        "--ctrl_tag",
        help="Choose an Enzymatic Commision (EC) control tag for finetuning ZymCTRL. Note that the tag must match all of the enzymes in the query fasta file. You can find all ECs here https://www.brenda-enzymes.org/ecexplorer.php?browser=1",
        action="store"
)
##############################################################################################################
    inv_fold = subparsers.add_parser('inv_fold_gen', help='Generate proteins using inverse folding')
    inv_fold.add_argument(
        "model",
        help="Choose between ESM-IF1 or ProteinMPNN to generate proteins using inverse folding.",
        choices = ['ESM-IF1', 'ProteinMPNN']
    )

    inv_fold.add_argument("query", 
        help="Input pdb file for inverse folding with ESM_IF1 or ProteinMPNN", 
        action="store"
        )

    inv_fold.add_argument(
        "--temp",
        help="Choose sampling temperature for ESM_IF1 or ProteinMPNN.",
        action="store",
        default = '1'
        )
    
    inv_fold.add_argument(
        "--num_return_sequences",
        help="Choose number of proteins for ESM-IF1 or ProteinMPNN to generate.",
        action="store",
        default = 1
        )
    
    inv_fold.add_argument(
        "--max_length",
        help="Max length of proteins generated from ESM-IF1 or ProteinMPNN",
        default=500,
        type=int
)

    inv_fold.add_argument("--mpnn_model", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")
    inv_fold.add_argument("--save_score", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True; save score=-log_prob to npy files")
    inv_fold.add_argument("--save_probs", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True; save MPNN predicted probabilites per position")
    inv_fold.add_argument("--score_only", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True; score input backbone-sequence pairs")
    inv_fold.add_argument("--path_to_fasta", type=str, default="", help="ProteinMPNN-only argument. score provided input sequence in a fasta format; e.g. GGGGGG/PPPPS/WWW for chains A, B, C sorted alphabetically and separated by /")
    inv_fold.add_argument("--conditional_probs_only", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)")    
    inv_fold.add_argument("--conditional_probs_only_backbone", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)") 
    inv_fold.add_argument("--unconditional_probs_only", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True; output unconditional probabilities p(s_i given backbone) in one forward pass")   
    inv_fold.add_argument("--backbone_noise", type=float, default=0.00, help="ProteinMPNN-only argument. Standard deviation of Gaussian noise to add to backbone atoms")
    inv_fold.add_argument("--batch_size", type=int, default=1, help="ProteinMPNN-only argument. Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    inv_fold.add_argument("--pdb_path_chains", type=str, default='', help="ProteinMPNN-only argument. Define which chains need to be designed for a single PDB ")
    inv_fold.add_argument("--chain_id_jsonl",type=str, default='', help="ProteinMPNN-only argument. Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    inv_fold.add_argument("--fixed_positions_jsonl", type=str, default='', help="ProteinMPNN-only argument. Path to a dictionary with fixed positions")
    inv_fold.add_argument("--omit_AAs", type=list, default='X', help="ProteinMPNN-only argument. Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    inv_fold.add_argument("--bias_AA_jsonl", type=str, default='', help="ProteinMPNN-only argument. Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
    inv_fold.add_argument("--bias_by_res_jsonl", default='', help="ProteinMPNN-only argument. Path to dictionary with per position bias.") 
    inv_fold.add_argument("--omit_AA_jsonl", type=str, default='', help="ProteinMPNN-only argument. Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
    inv_fold.add_argument("--pssm_jsonl", type=str, default='', help="ProteinMPNN-only argument. Path to a dictionary with pssm")
    inv_fold.add_argument("--pssm_multi", type=float, default=0.0, help="ProteinMPNN-only argument. A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
    inv_fold.add_argument("--pssm_threshold", type=float, default=0.0, help="ProteinMPNN-only argument. A value between -inf + inf to restric per position AAs")
    inv_fold.add_argument("--pssm_log_odds_flag", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True")
    inv_fold.add_argument("--pssm_bias_flag", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True")
    inv_fold.add_argument("--tied_positions_jsonl", type=str, default='', help="ProteinMPNN-only argument. Path to a dictionary with tied positions")

##############################################################################################################
    lang_gen = subparsers.add_parser('lang_gen', help='Generate proteins using large language models including ProtGPT2 and ESM2')

    lang_gen.add_argument(
        "model",
        help="Choose between Gibbs sampling with ESM2, ProtGPT2 or ZymCTRL.",
        choices = ['ESM2','ProtGPT2', 'ZymCTRL']
)
    lang_gen.add_argument(
        "--finetuned",
        help="Input path to your own finetuned ProtGPT2 or ESM2 model",
        action="store",
        default = False,
)
    lang_gen.add_argument(
        "--esm2_arch",
        help="Choose which ESM2 architecture your finetuned model is",
        action="store",
        default = 'esm2_t12_35M_UR50D',
)
    lang_gen.add_argument(
        "--temp",
        help="Choose sampling temperature.",
        action="store",
        default = '1',
)

    lang_gen.add_argument(
        "--ctrl_tag",
        help="Choose an Enzymatic Commision (EC) control tag for conditional protein generation based on the tag. You can find all ECs here https://www.brenda-enzymes.org/ecexplorer.php?browser=1",
        action="store",
)

    lang_gen.add_argument(
        "--seed_seq",
        help="Sequence to seed generation",
        default='M',
)
    lang_gen.add_argument(
        "--max_length",
        help="Max length of proteins generated",
        default=100,
        type=int
)
    lang_gen.add_argument(
        "--do_sample",
        help="Whether or not to use sampling for ProtGPT2 ; use greedy decoding otherwise",
        default=True,
        dest="do_sample",
)
    lang_gen.add_argument(
        "--top_k",
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering for ProtGPT2 or ESM2_Gibbs",
        default=950,
        dest="top_k",
        type=int
)
    lang_gen.add_argument(
        "--repetition_penalty",
        help="The parameter for repetition penalty for ProtGPT2. 1.0 means no penalty",
        default=1.2,
        dest="repetition_penalty",
)
    lang_gen.add_argument(
        "--num_return_sequences",
        help="Number of sequences for ProtGPT or ESM2_Gibbs to generate. Default is 5",
        default=5,
        dest="num_return_sequences",
        type=int,
)
    lang_gen.add_argument("--random_fill", 
        help="Randomly select positions to fill each iteration for Gibbs sampling with ESM2. If not called then fill the positions in order", 
        action="store_false",
        default = True,
        )
    lang_gen.add_argument("--num_positions", 
        help="Generate new AAs for this many positions each iteration for Gibbs sampling with ESM2. If 0, then generate for all target positions each round.", 
        action="store",
        default = 0,
        )
    
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
        default=None
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
    classify = subparsers.add_parser('classify', help='Classify proteins based on thermostability predicted through TemStaPro')

    classify.add_argument(
        "classifier",
        help="Predict thermostability using TemStaPro or choose custom to train/use your own XGBoost based binary classifier. Note for training a custom_binary, you need to submit roughly equal amounts of both binary classes as part of your query.",
        choices = ['TemStaPro', 'custom_binary']
)
    classify.add_argument(
        "query",
        help="Fasta file of sequences to score",
        action="store"
)
    classify.add_argument(
        "--key",
        help="String that allows for the unique identification of your binary classes from the input fasta headers. For example, --key positive_hits would group all sequences that have 'positive_hits' in the fasta header as one class and the rest as the other class",
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
        help="Select between 'esm2_t6_8M', 'esm2_t12_35M', 'esm2_t30_150M', 'esm2_t33_650M', 'esm2_t36_3B','esm2_t48_15B', or 'ProtT5-XL' for embedding your query proteins to then train your custom classifier",
        default = 'esm2_t12_35M',
        action="store"
)
    classify.add_argument(
        "--train_split",
        help="Choose your train-test percentage split for training and evaluating your custom classifier. For example, --train .6 would split your input sequences into two groups, one with 60%% of the sequences to train and the other with 40%% for evaluating",
        action="store",
)
    classify.add_argument(
        "--preTrained",
        help="Enter the path to your pre-trained XGBoost binary classifier that you've trained with TRILL.",
        action="store",
)
##############################################################################################################
    
    fold = subparsers.add_parser('fold', help='Predict 3D protein structures using ESMFold')

    fold.add_argument("query", 
        help="Input fasta file", 
        action="store"
        )
    fold.add_argument("--strategy", 
        help="Choose a specific strategy if you are running out of CUDA memory. You can also pass either 64, or 32 for model.trunk.set_chunk_size(x)", 
        action="store",
        default = None,
        )    
##############################################################################################################
    visualize = subparsers.add_parser('visualize', help='Reduce dimensionality of embeddings to 2D')

    visualize.add_argument("embeddings", 
        help="Embeddings to be visualized", 
        action="store"
        )
    
    visualize.add_argument("--method", 
        help="Method for reducing dimensions of embeddings. Default is PCA, but you can also choose UMAP or tSNE", 
        action="store",
        default="PCA"
        )
    visualize.add_argument("--group", 
        help="Grouping for color scheme of output scatterplot. Choose this option if the labels in your embedding csv are grouped by the last pattern separated by an underscore. For example, 'Protein1_group1', 'Protein2_group1', 'Protein3_group2'. By default, all points are treated as same group.", 
        action="store_true",
        default=False
        )
    
##############################################################################################################
    dock = subparsers.add_parser('dock', help='Dock protein to protein using DiffDock')

    dock.add_argument("protein", 
        help="Protein of interest to be docked with ligand", 
        action="store"
        )
    
    dock.add_argument("ligand", 
        help="Ligand to dock protein with", 
        action="store",
        )
    dock.add_argument("--save_visualisation", 
        help="Save a pdb file with all of the steps of the reverse diffusion.", 
        action="store_true",
        default=False
        )
    
    dock.add_argument("--samples_per_complex", 
        help="Number of samples to generate.", 
        type = int,
        action="store",
        default=10
        )
    
    dock.add_argument("--no_final_step_noise", 
        help="Use no noise in the final step of the reverse diffusion", 
        action="store_true",
        default=False
        )
    
    dock.add_argument("--inference_steps", 
        help="Number of denoising steps", 
        type=int,
        action="store",
        default=20
        )

    dock.add_argument("--actual_steps", 
        help="Number of denoising steps that are actually performed", 
        type=int,
        action="store",
        default=None
        )

##############################################################################################################

    

    

    args = parser.parse_args()

    pl.seed_everything(int(args.RNG_seed))
    set_seed(int(args.RNG_seed))
    
    
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

    if args.command == 'visualize':
        reduced_df, incsv = reduce_dims(args.name, args.embeddings, args.method)
        fig = viz(reduced_df, args.name, args.group)
        bokeh.io.output_file(filename=f'{args.name}_{args.method}_{incsv}.html', title=args.name) 
        bokeh.io.save(fig, filename=f'{args.name}_{args.method}_{incsv}.html', title = args.name)
    

    elif args.command == 'embed':
        if args.query.endswith(('.fasta', '.faa', '.fa')) == False:
            raise Exception(f'Input query file - {args.query} is not a valid file format.\
            File needs to be a protein fasta (.fa, .fasta, .faa)')
        if args.model == "ProtT5-XL":
            model = ProtT5(args)
            data = esm.data.FastaBatchedDataset.from_file(args.query)
            dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = int(args.batch_size), num_workers=0)
            pred_writer = CustomWriter(output_dir=".", write_interval="epoch")
            if int(args.GPUs) == 0:
                trainer = pl.Trainer(enable_checkpointing=False, callbacks = [pred_writer], logger=logger, num_nodes=int(args.nodes))
            else:
                trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), callbacks = [pred_writer], accelerator='gpu', logger=logger, num_nodes=int(args.nodes))
            trainer.predict(model, dataloader)
            cwd_files = os.listdir()
            pt_files = [file for file in cwd_files if 'predictions_' in file]
            pred_embeddings = []
            for pt in pt_files:
                preds = torch.load(pt)
                for pred in preds:
                    for sublist in pred:
                        if len(sublist) == 2:
                            pred_embeddings.append(tuple([sublist[0], sublist[1]]))
                        else:
                            for sub in sublist:
                                print(sub[0])
                                print(sub[1])
                        #     for sub in sublist:
                        #         pred_embeddings.append(tuple([sub[0], sub[1]]))
            embedding_df = pd.DataFrame(pred_embeddings, columns = ['Embeddings', 'Label'])
            finaldf = embedding_df['Embeddings'].apply(pd.Series)
            finaldf['Label'] = embedding_df['Label']
            finaldf.to_csv(f'{args.name}_{args.model}.csv', index = False)
            for file in pt_files:
                os.remove(file)
        else:
            model_import_name = f'esm.pretrained.{args.model}_UR50D()'
            model = ESM(eval(model_import_name), 0.0001, args, False)
            data = esm.data.FastaBatchedDataset.from_file(args.query)
            dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = int(args.batch_size), num_workers=0, collate_fn=model.alphabet.get_batch_converter())
            pred_writer = CustomWriter(output_dir=".", write_interval="epoch")
            if int(args.GPUs) == 0:
                trainer = pl.Trainer(enable_checkpointing=False, callbacks = [pred_writer], logger=logger, num_nodes=int(args.nodes))
            else:
                trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), callbacks = [pred_writer], accelerator='gpu', logger=logger, num_nodes=int(args.nodes))
            if args.finetuned == False:
                trainer.predict(model, dataloader)
                cwd_files = os.listdir()
                pt_files = [file for file in cwd_files if 'predictions_' in file]
                pred_embeddings = []
                for pt in pt_files:
                    preds = torch.load(pt)
                    for pred in preds:
                        for sublist in pred:
                            if len(sublist) == 1:
                                pred_embeddings.append(tuple([sublist[0][0], sublist[0][1]]))
                            else:
                                for sub in sublist:
                                    pred_embeddings.append(tuple([sub[0], sub[1]]))
                embedding_df = pd.DataFrame(pred_embeddings, columns = ['Embeddings', 'Label'])
                finaldf = embedding_df['Embeddings'].apply(pd.Series)
                finaldf['Label'] = embedding_df['Label']
                finaldf.to_csv(f'{args.name}_{args.model}.csv', index = False)
                for file in pt_files:
                    os.remove(file)


            else:
                model = weights_update(model = ESM(eval(model_import_name), 0.0001, args, False), checkpoint = torch.load(args.finetuned))
                trainer.predict(model, dataloader)
                cwd_files = os.listdir()
                pt_files = [file for file in cwd_files if 'predictions_' in file]
                pred_embeddings = []
                for pt in pt_files:
                    preds = torch.load(pt)
                    for pred in preds:
                        for sublist in pred:
                            pred_embeddings.append(tuple([sublist[0][0], sublist[0][1]]))
                embedding_df = pd.DataFrame(pred_embeddings, columns = ['Embeddings', 'Label'])
                finaldf = embedding_df['Embeddings'].apply(pd.Series)
                finaldf['Label'] = embedding_df['Label']
                finaldf.to_csv(f'{args.name}_{args.model}.csv', index = False)
                for file in pt_files:
                    os.remove(file)

    
    elif args.command == 'finetune':
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        len_data = len(data)
        if args.model == 'ProtGPT2':
            model = ProtGPT2(args)
            tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
            seq_dict_df = ProtGPT2_wrangle(data, tokenizer)
            dataloader = torch.utils.data.DataLoader(seq_dict_df, shuffle = False, batch_size = int(args.batch_size), num_workers=0)
            if args.save_on_epoch:
                checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k = -1)
                if int(args.GPUs) == 0:
                    trainer = pl.Trainer(profiler=profiler, max_epochs=int(args.epochs), logger = logger, num_nodes = int(args.nodes), callbacks=[checkpoint_callback], default_root_dir=f'{os.path.join(os.getcwd(), args.name)}_ckpt')
                else:
                    trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator='gpu', max_epochs=int(args.epochs), logger = logger, num_nodes = int(args.nodes), precision = 16, strategy = args.strategy, callbacks=[checkpoint_callback], default_root_dir=f'{os.path.join(os.getcwd(), args.name)}_ckpt')
            else:
                if int(args.GPUs) == 0:
                    trainer = pl.Trainer(profiler=profiler, max_epochs=int(args.epochs), logger = logger, num_nodes = int(args.nodes), enable_checkpointing=False)
                else:
                    trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator='gpu', default_root_dir=f'{os.path.join(os.getcwd(), args.name)}_ckpt', max_epochs=int(args.epochs), logger = logger, num_nodes = int(args.nodes), precision = 16, strategy = args.strategy)
            trainer.fit(model=model, train_dataloaders = dataloader)
            if 'deepspeed' in str(args.strategy):
                save_path = os.path.join(os.getcwd(), f"{os.path.join(os.getcwd(), args.name)}_ckpt/checkpoints/epoch={int(args.epochs) - 1}-step={len_data*int(args.epochs)}.ckpt")
                output_path = f"{args.name}_ProtGPT2_{args.epochs}.pt"
                try:
                    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
                except Exception as e:
                    print(f'Exception {e} has occured on attempted save of your deepspeed trained model. If this has to do with CPU RAM, please try pytorch_lightning.utilities.deepspeedconvert_zero_checkpoint_to_fp32_state_dict(your_checkpoint.ckpt, full_model.pt')
            elif str(args.strategy) in ['fsdp', 'FSDP', 'FullyShardedDataParallel']:
                pass

            else:
                trainer.save_checkpoint(f"{args.name}_{args.model}_{args.epochs}.pt")
        elif args.model == 'ZymCTRL':
            model = ZymCTRL(args)
            seq_dict_df = ProtGPT2_wrangle(data, model.tokenizer)
            dataloader = torch.utils.data.DataLoader(seq_dict_df, shuffle = False, batch_size = int(args.batch_size), num_workers=0)
            if args.save_on_epoch:
                checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k = -1)
                if int(args.GPUs) == 0:
                    trainer = pl.Trainer(profiler=profiler, max_epochs=int(args.epochs), logger = logger, num_nodes = int(args.nodes), callbacks=[checkpoint_callback], default_root_dir=f'{os.path.join(os.getcwd(), args.name)}_ckpt')
                else:
                    trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator='gpu', max_epochs=int(args.epochs), logger = logger, num_nodes = int(args.nodes), precision = 16, strategy = args.strategy, callbacks=[checkpoint_callback], default_root_dir=f'{os.path.join(os.getcwd(), args.name)}_ckpt')
            else:
                if int(args.GPUs) == 0:
                    trainer = pl.Trainer(profiler=profiler, max_epochs=int(args.epochs), logger = logger, num_nodes = int(args.nodes), enable_checkpointing=False)
                else:
                    trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator='gpu', default_root_dir=f'{os.path.join(os.getcwd(), args.name)}_ckpt', max_epochs=int(args.epochs), logger = logger, num_nodes = int(args.nodes), precision = 16, strategy = args.strategy)
            trainer.fit(model=model, train_dataloaders = dataloader)
            if 'deepspeed' in str(args.strategy):
                save_path = os.path.join(os.getcwd(), f"{os.path.join(os.getcwd(), args.name)}_ckpt/checkpoints/epoch={int(args.epochs) - 1}-step={len_data*int(args.epochs)}.ckpt")
                output_path = f"{args.name}_ZymCTRL_{args.epochs}.pt"
                try:
                    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
                except Exception as e:
                    print(f'Exception {e} has occured on attempted save of your deepspeed trained model. If this has to do with CPU RAM, please try pytorch_lightning.utilities.deepspeedconvert_zero_checkpoint_to_fp32_state_dict(your_checkpoint.ckpt, full_model.pt')
            elif str(args.strategy) in ['fsdp', 'FSDP', 'FullyShardedDataParallel']:
                pass

        else:
            model_import_name = f'esm.pretrained.{args.model}_UR50D()'
            model = ESM(eval(model_import_name), float(args.lr), args)
            dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = int(args.batch_size), num_workers=0, collate_fn=model.alphabet.get_batch_converter())
            # if args.LEGGO:
            #     raise RuntimeError('LEGGO is no longer a valid option. Sorry for the confusion')
            #     trainer = pl.Trainer(devices=int(args.GPUs), profiler = profiler,accelerator='gpu',max_epochs=int(args.epochs),logger=logger, num_nodes=int(args.nodes), precision = 16, strategy=DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True))
            #     trainer.fit(model=model, train_dataloaders=dataloader)
            #     save_path = os.path.join(os.getcwd(), f"checkpoints/epoch={int(args.epochs) - 1}-step={len_data*int(args.epochs)}.ckpt")
            #     output_path = f"{args.name}_esm2_{args.epochs}.pt"
            #     try:
            #         convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
            #     except Exception as e:
            #         print(f'Exception {e} has occured on attempted save of your deepspeed trained model. If this has to do with CPU RAM, please try pytorch_lightning.utilities.deepspeedconvert_zero_checkpoint_to_fp32_state_dict(your_checkpoint.ckpt, full_model.pt')       
            if args.strategy == 'deepspeed_stage_3' or args.strategy == 'deepspeed_stage_3_offload' or args.strategy == 'deepspeed_stage_2' or args.strategy == 'deepspeed_stage_2_offload':
                save_path = os.path.join(os.getcwd(), f"checkpoints/epoch={int(args.epochs) - 1}-step={len_data*int(args.epochs)}.ckpt")
                output_path = f"{args.name}_esm2_{args.epochs}.pt"
                if args.save_on_epoch:
                    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k = -1)
                    trainer = pl.Trainer(devices=int(args.GPUs), profiler = profiler, callbacks=[checkpoint_callback], default_root_dir=f'{os.path.join(os.getcwd(), args.name)}_ckpt', accelerator='gpu', strategy = args.strategy, max_epochs=int(args.epochs), logger=logger, num_nodes=int(args.nodes), precision = 16)        
                else:
                    trainer = pl.Trainer(devices=int(args.GPUs), profiler = profiler, callbacks=[checkpoint_callback], default_root_dir=f'{os.path.join(os.getcwd(), args.name)}_ckpt', accelerator='gpu', strategy = args.strategy, max_epochs=int(args.epochs), logger=logger, num_nodes=int(args.nodes), precision = 16)        
                trainer.fit(model=model, train_dataloaders=dataloader)
                trainer.save_checkpoint(f"{args.name}_{args.model}_{args.epochs}.pt")
                try:
                    convert_zero_checkpoint_to_fp32_state_dict(f"{args.name}_{args.model}_{args.epochs}.pt", output_path)
                except Exception as e:
                    print(f'Exception {e} has occured on attempted save of your deepspeed trained model. If this has to do with CPU RAM, please try pytorch_lightning.utilities.deepspeedconvert_zero_checkpoint_to_fp32_state_dict(your_checkpoint.ckpt, full_model.pt')       
            else:
                if args.save_on_epoch:
                    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k = -1)
                    if int(args.GPUs) == 0:
                        trainer = pl.Trainer(profiler = profiler, max_epochs=int(args.epochs), callbacks=[checkpoint_callback], default_root_dir=f'{os.path.join(os.getcwd(), args.name)}_ckpt', logger=logger, num_nodes=int(args.nodes)) 
                    else:
                        trainer = pl.Trainer(devices=int(args.GPUs), profiler = profiler, accelerator='gpu', callbacks=[checkpoint_callback], default_root_dir=f'{os.path.join(os.getcwd(), args.name)}_ckpt',strategy = args.strategy, max_epochs=int(args.epochs), logger=logger, num_nodes=int(args.nodes), precision = 16, amp_backend='native')        
                else:
                    if int(args.GPUs) == 0:
                        trainer = pl.Trainer(profiler = profiler, max_epochs=int(args.epochs), logger=logger, num_nodes=int(args.nodes), enable_checkpointing=False) 
                    else:
                        trainer = pl.Trainer(devices=int(args.GPUs), profiler = profiler, accelerator='gpu', strategy = args.strategy, max_epochs=int(args.epochs), logger=logger, num_nodes=int(args.nodes), precision = 16, amp_backend='native', enable_checkpointing=False)     
                trainer.fit(model=model, train_dataloaders=dataloader)
                trainer.save_checkpoint(f"{args.name}_{args.model}_{args.epochs}.pt")

    elif args.command == 'inv_fold_gen':
        if args.model == 'ESM-IF1':
            if args.query == None:
                raise Exception('A PDB or CIF file is needed for generating new proteins with ESM-IF1')
            data = ESM_IF1_Wrangle(args.query)
            dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
            sample_df, native_seq_df = ESM_IF1(dataloader, genIters=int(args.num_return_sequences), temp = float(args.temp), GPUs = int(args.GPUs))
            pdb_name = args.query.split('.')[-2].split('/')[-1]
            with open(f'{args.name}_ESM-IF1_gen.fasta', 'w+') as fasta:
                for ix, row in native_seq_df.iterrows():
                    fasta.write(f'>{pdb_name}_chain-{row[1]} \n')
                    fasta.write(f'{row[0][0]}\n')
                for ix, row in sample_df.iterrows():
                    fasta.write(f'>{args.name}_ESM-IF1_chain-{row[1]} \n')
                    fasta.write(f'{row[0]}\n')
        elif args.model == 'ProteinMPNN':
            if not os.path.exists('ProteinMPNN/'):
                print('Cloning forked ProteinMPNN')
                os.makedirs('ProteinMPNN/')
                proteinmpnn = Repo.clone_from('https://github.com/martinez-zacharya/ProteinMPNN', 'ProteinMPNN/')
                mpnn_git_root = proteinmpnn.git.rev_parse("--show-toplevel")
                subprocess.run(['pip', 'install', '-e', mpnn_git_root])
                sys.path.insert(0, 'ProteinMPNN/')
            else:
                sys.path.insert(0, 'ProteinMPNN/')
            from mpnnrun import run_mpnn
            print('ProteinMPNN generation starting...')
            run_mpnn(args)
        
    elif args.command == 'lang_gen':
        if args.model == 'ProtGPT2':
            model = ProtGPT2(args)
            if args.finetuned != False:
                model = model.load_from_checkpoint(args.finetuned, args = args, strict=False)
            tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
            generated_output = []
            with open(f'{args.name}_ProtGPT2.fasta', 'w+') as fasta:
                for i in tqdm(range(int(args.num_return_sequences))):
                    generated_output = (model.generate(seed_seq=args.seed_seq, max_length=int(args.max_length), do_sample = args.do_sample, top_k=int(args.top_k), repetition_penalty=float(args.repetition_penalty)))
                    fasta.write(f'>{args.name}_ProtGPT2_{i} \n')
                    fasta.write(f'{generated_output[0]}\n')
                    fasta.flush()
        elif args.model == 'ESM2':
            model_import_name = f'esm.pretrained.{args.esm2_arch}()'
            with open(f'{args.name}_{args.esm2_arch}_Gibbs.fasta', 'w+') as fasta:
                if args.finetuned != False:
                    model = ESM_Gibbs(eval(model_import_name), args)
                    if args.finetuned != False:
                        model = weights_update(model = ESM_Gibbs(eval(model_import_name), args), checkpoint = torch.load(args.finetuned))
                        tuned_name = args.finetuned.split('/')[-1] 
                    if int(args.GPUs) > 0:
                        model.model = model.model.cuda()
                    for i in range(args.num_return_sequences):
                        out = model.generate(args.seed_seq, mask=True, n_samples = 1, max_len = args.max_length, in_order = args.random_fill, num_positions=int(args.num_positions), temperature=float(args.temp))
                        out = ''.join(out)
                        fasta.write(f'>{args.name}_{tuned_name[0:-3]}_Gibbs_{i} \n')
                        fasta.write(f'{out}\n')
                        fasta.flush()           
                else:
                    model = ESM_Gibbs(eval(model_import_name), args)
                    tuned_name = f'{args.esm2_arch}___'
                    if int(args.GPUs) > 0:
                        model.model = model.model.cuda()
                    for i in range(args.num_return_sequences):
                        out = model.generate(args.seed_seq, mask=True, n_samples = 1, max_len = args.max_length, in_order = args.random_fill, num_positions=int(args.num_positions), temperature=float(args.temp))
                        out = ''.join(out)
                        fasta.write(f'>{args.name}_{tuned_name[0:-3]}_Gibbs_{i} \n')
                        fasta.write(f'{out}\n')
                        fasta.flush()  

        elif args.model == 'ZymCTRL':
            model = ZymCTRL(args)
            if args.finetuned != False:
                model = model.load_from_checkpoint(args.finetuned, args = args, strict = False)
            with open(f'{args.name}_ZymCTRL.fasta', 'w+') as fasta:
                for i in tqdm(range(int(args.num_return_sequences))):
                    if int(args.GPUs) == 0:
                        generated_output = model.generator(str(args.ctrl_tag), device = torch.device('cpu'), max_length=int(args.max_length),repetition_penalty=float(args.repetition_penalty), do_sample=args.do_sample, top_k=int(args.top_k))
                    else:
                        generated_output = model.generator(str(args.ctrl_tag), device = torch.device('cuda'), max_length=int(args.max_length),repetition_penalty=float(args.repetition_penalty), do_sample=args.do_sample, top_k=int(args.top_k))
                    fasta.write(f'>{args.name}_{args.ctrl_tag}_ZymCTRL_{i}_PPL={generated_output[0][1]} \n')
                    fasta.write(f'{generated_output[0][0]}\n')
                    fasta.flush()
                    
    elif args.command == 'diff_gen':
        command = "conda install -c dglteam dgl-cuda11.7 -y -S -q".split(' ')
        subprocess.run(command, check = True)
        os.makedirs("RFDiffusion_weights", exist_ok=True)
        commands = [
        'wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt', 
        'wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt', 
        'wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt', 
        'wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt', 
        'wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt', 
        'wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt', 
        'wget -nc http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt'
        ]

        print('Finding RFDiffusion weights... \n')
        for command in commands:
            if not os.path.isfile(f'RFDiffusion_weights/{command.split("/")[-1]}'):
                subprocess.run(command.split(' '))
                subprocess.run(['mv', command.split("/")[-1], 'RFDiffusion_weights/'])
        if not os.path.exists('RFDiffusion/'):
            print('Cloning forked RFDiffusion')
            os.makedirs('RFDiffusion/')
            rfdiff = Repo.clone_from('https://github.com/martinez-zacharya/RFDiffusion', 'RFDiffusion/')
            rfdiff_git_root = rfdiff.git.rev_parse("--show-toplevel")
            subprocess.run(['pip', 'install', '-e', rfdiff_git_root])
            command = f'pip install {rfdiff_git_root}/env/SE3Transformer'.split(' ')
            subprocess.run(command)
            sys.path.insert(0, 'RFDiffusion/')

        else:
            sys.path.insert(0, 'RFDiffusion/')
            git_repo = Repo('RFDiffusion/', search_parent_directories=True)
            rfdiff_git_root = git_repo.git.rev_parse("--show-toplevel")

        from run_inference import run_rfdiff
        # if args.sym:
        #     run_rfdiff((f'{rfdiff_git_root}/config/inference/symmetry.yaml'), args)
        # else:    
        #     run_rfdiff((f'{rfdiff_git_root}/config/inference/base.yaml'), args)
        run_rfdiff((f'{rfdiff_git_root}/config/inference/base.yaml'), args)
            
    elif args.command == 'fold':
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

        if int(args.GPUs) == 0:
            model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1', low_cpu_mem_usage=True, torch_dtype='auto')
        else:
            model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1', device_map='auto', torch_dtype='auto')
            model.esm = model.esm.half()
            device = torch.device("cuda")
            model = model.to(device)
        if args.strategy != None:
            model.trunk.set_chunk_size(int(args.strategy))
        fold_df = pd.DataFrame(list(data), columns = ["Entry", "Sequence"])
        outputs = []
        with torch.no_grad():
            for input_ids in tqdm(fold_df.Sequence.tolist()):
                tokenized_input = tokenizer([input_ids], return_tensors="pt", add_special_tokens=False)['input_ids']
                tokenized_input = tokenized_input.clone().detach()
                prot_len = len(input_ids)
                try:
                    output = model(tokenized_input)
                    outputs.append({key: val.cpu() for key, val in output.items()})
                except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print(f'Protein too long to fold for current hardware: {prot_len} amino acids long)')
                            print(e)
                        else:
                            print(e)
                            pass

        pdb_list = [convert_outputs_to_pdb(output) for output in outputs]
        protein_identifiers = fold_df.Entry.tolist()
        for identifier, pdb in zip(protein_identifiers, pdb_list):
            with open(f"{identifier}.pdb", "w") as f:
                f.write("".join(pdb))

    elif args.command == 'dock':
        if not os.path.exists('DiffDock/'):
            print('Cloning forked DiffDock')
            os.makedirs('DiffDock/')
            diffdock = Repo.clone_from('https://github.com/martinez-zacharya/DiffDock', 'DiffDock/')
            diffdock_root = diffdock.git.rev_parse("--show-toplevel")
            subprocess.run(['pip', 'install', '-e', diffdock_root])
            sys.path.insert(0, 'DiffDock/')
        else:
            sys.path.insert(0, 'DiffDock/')
            diffdock = Repo('DiffDock')
            diffdock_root = diffdock.git.rev_parse("--show-toplevel")
        from inference import run_diffdock
        run_diffdock(args, diffdock_root)

    elif args.command == 'classify':
        if args.classifier == 'TemStaPro':
            data = esm.data.FastaBatchedDataset.from_file(args.query)
            model = ProtT5(args)
            dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = 1, num_workers=0)
            if int(args.GPUs) > 0:
                trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), accelerator='gpu', logger=logger, num_nodes=int(args.nodes))
            else:
                trainer = pl.Trainer(enable_checkpointing=False, logger=logger, num_nodes=int(args.nodes))
            reps = trainer.predict(model, dataloader)
            if args.save_emb:
                emb_4export = []
                for rep in reps:
                    intermed = []
                    for i in range(len(rep[0])):
                        intermed.append(rep[0][i].cpu().numpy())
                    intermed.append(rep[1])
                    emb_4export.append(intermed)
                emb_df = pd.DataFrame(emb_4export)
                emb_df.to_csv(f'{args.name}_ProtT5-XL_embeddings.csv', index=False)
            if not os.path.exists('TemStaPro_models/'):
                temstapro_models = Repo.clone_from('https://github.com/martinez-zacharya/TemStaPro_models', 'TemStaPro_models/')
                temstapro_models_root = temstapro_models.git.rev_parse("--show-toplevel")
            else:
                temstapro_models = Repo('TemStaPro_models')
                temstapro_models_root = temstapro_models.git.rev_parse("--show-toplevel")
            dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = 1, num_workers=0)
            THRESHOLDS = ["40", "45", "50", "55", "60", "65"]
            SEEDS = ["41", "42", "43", "44", "45"]
            emb_loader = torch.utils.data.DataLoader(reps, shuffle = False, batch_size = 1, num_workers = 0)
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
                    binary_pred = round(mean_prediction)
                    inferences[f'{seq}$%#{thresh}'] = (mean_prediction, binary_pred)
            inference_df = pd.DataFrame.from_dict(inferences, orient='index', columns=['Mean_Pred', 'Binary_Pred'])
            inference_df = inference_df.reset_index(names='RawLab')
            inference_df['Protein'] = inference_df['RawLab'].apply(lambda x: x.split('$%#')[0])
            inference_df['Threshold'] = inference_df['RawLab'].apply(lambda x: x.split('$%#')[-1])
            inference_df = inference_df.drop(columns='RawLab')
            inference_df = inference_df[['Protein', 'Threshold', 'Mean_Pred', 'Binary_Pred']]
            inference_df.to_csv(f'{args.name}_TemStaPro_preds.csv', index = False)
        elif args.classifier == 'custom_binary':
            embed_command = f"trill {args.name} {args.GPUs} embed {args.emb_model} {args.query}".split(' ')
            subprocess.run(embed_command, check=True)
            df = pd.read_csv(f'{args.name}_{args.emb_model}.csv')
            if args.train_split is not None:
                df['NewLab'] = np.where(df['Label'].str.contains(args.key) == 1, 1, 0)
                df = df.sample(frac = 1)
                train_df, test_df = train_test_split(df, test_size = float(args.train_split), stratify = df['NewLab'])
                model = xgb.XGBClassifier(gamma= 0.4, learning_rate = 0.2,  max_depth = 8, n_estimators = 115, reg_alpha = 0.8, reg_lambda = 0.1)
                model.fit(train_df.iloc[:,:-2], train_df['NewLab'])
                test_preds = model.predict(test_df.iloc[:,:-2])
                precision, recall, fscore, support = precision_recall_fscore_support(test_df['NewLab'], test_preds, average = 'binary')
                print(f'{precision=}')
                print(f'{recall=}')
                print(f'{fscore=}')

                if not args.save_emb:
                    os.remove(f'{args.name}_{args.emb_model}.csv')
                model.save_model(f'{args.name}_XGBoost_binary_{len(test_df.columns)-2}.json')
            else:
                model = xgb.Booster()
                model.load_model(args.preTrained)
                data = xgb.DMatrix(df.iloc[:,:-1])
                test_preds = model.predict(data)
                binary_scores = test_preds.round()
                df['Predicted_Class'] = binary_scores.astype(int)
                out_df = df[['Label', 'Predicted_Class']]
                out_df.to_csv(f'{args.name}_predictions.csv', index = False)
                if not args.save_emb:
                    os.remove(f'{args.name}_{args.emb_model}.csv')






        



    
    end = time.time()
    print("Finished!")
    print(f"Time elapsed: {end-start} seconds")
 

def cli(args=None):
    if not args:
        args = sys.argv[1:]    
    main(args)
if __name__ == '__main__':
    print("this shouldn't show up...")

    

    
def return_parser():
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


##############################################################################################################

    subparsers = parser.add_subparsers(dest='command')

    embed = subparsers.add_parser('embed', help='Embed proteins of interest')

    embed.add_argument(
        "model",
        help="You can choose from either 'esm2_t6_8M', 'esm2_t12_35M', 'esm2_t30_150M', 'esm2_t33_650M', 'esm2_t36_3B','esm2_t48_15B', or 'ProtT5-XL'",
        action="store",
        # default = 'esm2_t12_35M_UR50D',
)

    embed.add_argument("query", 
        help="Input fasta file", 
        action="store"
)
    embed.add_argument(
        "--batch_size",
        help="Change batch-size number for embedding proteins. Default is 1",
        action="store",
        default = 1,
        dest="batch_size",
)

    embed.add_argument(
        "--finetuned",
        help="Input path to your own finetuned ESM model",
        action="store",
        default = False,
        dest="finetuned",
)
##############################################################################################################

    finetune = subparsers.add_parser('finetune', help='Finetune protein language models')

    finetune.add_argument(
        "model",
        help="You can choose to finetune either 'esm2_t6_8M', 'esm2_t12_35M', 'esm2_t30_150M', 'esm2_t33_650M', 'esm2_t36_3B','esm2_t48_15B', 'ProtGPT2', or ZymCTRL.",
        action="store",
)

    finetune.add_argument("query", 
        help="Input fasta file", 
        action="store"
)
    finetune.add_argument("--epochs", 
        help="Number of epochs for fine-tuning. Default is 20", 
        action="store",
        default=10,
        dest="epochs",
        )
    finetune.add_argument("--save_on_epoch", 
        help="Saves a checkpoint on every successful epoch completed. WARNING, this could lead to rapid storage consumption", 
        action="store_true",
        default=False,
        )
    finetune.add_argument(
        "--lr",
        help="Learning rate for optimizer. Default is 0.0001",
        action="store",
        default=0.0001,
        dest="lr",
)

    finetune.add_argument(
        "--batch_size",
        help="Change batch-size number for fine-tuning. Default is 1",
        action="store",
        default = 1,
        dest="batch_size",
)

    finetune.add_argument(
        "--strategy",
        help="Change training strategy. Default is None. List of strategies can be found at https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html",
        action="store",
        default = None,
        dest="strategy",
)

    finetune.add_argument(
        "--ctrl_tag",
        help="Choose an Enzymatic Commision (EC) control tag for finetuning ZymCTRL. Note that the tag must match all of the enzymes in the query fasta file. You can find all ECs here https://www.brenda-enzymes.org/ecexplorer.php?browser=1",
        action="store"
)
##############################################################################################################
    inv_fold = subparsers.add_parser('inv_fold_gen', help='Generate proteins using inverse folding')
    inv_fold.add_argument(
        "model",
        help="Choose between ESM-IF1 or ProteinMPNN to generate proteins using inverse folding.",
        choices = ['ESM-IF1', 'ProteinMPNN']
    )

    inv_fold.add_argument("query", 
        help="Input pdb file for inverse folding with ESM_IF1 or ProteinMPNN", 
        action="store"
        )

    inv_fold.add_argument(
        "--temp",
        help="Choose sampling temperature for ESM_IF1 or ProteinMPNN.",
        action="store",
        default = '1'
        )
    
    inv_fold.add_argument(
        "--num_return_sequences",
        help="Choose number of proteins for ESM-IF1 or ProteinMPNN to generate.",
        action="store",
        default = 1
        )
    
    inv_fold.add_argument(
        "--max_length",
        help="Max length of proteins generated from ESM-IF1 or ProteinMPNN",
        default=500,
        type=int
)

    inv_fold.add_argument("--mpnn_model", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")
    inv_fold.add_argument("--save_score", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True; save score=-log_prob to npy files")
    inv_fold.add_argument("--save_probs", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True; save MPNN predicted probabilites per position")
    inv_fold.add_argument("--score_only", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True; score input backbone-sequence pairs")
    inv_fold.add_argument("--path_to_fasta", type=str, default="", help="ProteinMPNN-only argument. score provided input sequence in a fasta format; e.g. GGGGGG/PPPPS/WWW for chains A, B, C sorted alphabetically and separated by /")
    inv_fold.add_argument("--conditional_probs_only", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)")    
    inv_fold.add_argument("--conditional_probs_only_backbone", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)") 
    inv_fold.add_argument("--unconditional_probs_only", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True; output unconditional probabilities p(s_i given backbone) in one forward pass")   
    inv_fold.add_argument("--backbone_noise", type=float, default=0.00, help="ProteinMPNN-only argument. Standard deviation of Gaussian noise to add to backbone atoms")
    inv_fold.add_argument("--batch_size", type=int, default=1, help="ProteinMPNN-only argument. Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    inv_fold.add_argument("--pdb_path_chains", type=str, default='', help="ProteinMPNN-only argument. Define which chains need to be designed for a single PDB ")
    inv_fold.add_argument("--chain_id_jsonl",type=str, default='', help="ProteinMPNN-only argument. Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    inv_fold.add_argument("--fixed_positions_jsonl", type=str, default='', help="ProteinMPNN-only argument. Path to a dictionary with fixed positions")
    inv_fold.add_argument("--omit_AAs", type=list, default='X', help="ProteinMPNN-only argument. Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    inv_fold.add_argument("--bias_AA_jsonl", type=str, default='', help="ProteinMPNN-only argument. Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
    inv_fold.add_argument("--bias_by_res_jsonl", default='', help="ProteinMPNN-only argument. Path to dictionary with per position bias.") 
    inv_fold.add_argument("--omit_AA_jsonl", type=str, default='', help="ProteinMPNN-only argument. Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
    inv_fold.add_argument("--pssm_jsonl", type=str, default='', help="ProteinMPNN-only argument. Path to a dictionary with pssm")
    inv_fold.add_argument("--pssm_multi", type=float, default=0.0, help="ProteinMPNN-only argument. A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
    inv_fold.add_argument("--pssm_threshold", type=float, default=0.0, help="ProteinMPNN-only argument. A value between -inf + inf to restric per position AAs")
    inv_fold.add_argument("--pssm_log_odds_flag", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True")
    inv_fold.add_argument("--pssm_bias_flag", type=int, default=0, help="ProteinMPNN-only argument. 0 for False, 1 for True")
    inv_fold.add_argument("--tied_positions_jsonl", type=str, default='', help="ProteinMPNN-only argument. Path to a dictionary with tied positions")

##############################################################################################################
    lang_gen = subparsers.add_parser('lang_gen', help='Generate proteins using large language models including ProtGPT2 and ESM2')

    lang_gen.add_argument(
        "model",
        help="Choose between Gibbs sampling with ESM2, ProtGPT2 or ZymCTRL.",
        choices = ['ESM2','ProtGPT2', 'ZymCTRL']
)
    lang_gen.add_argument(
        "--finetuned",
        help="Input path to your own finetuned ProtGPT2 or ESM2 model",
        action="store",
        default = False,
)
    lang_gen.add_argument(
        "--esm2_arch",
        help="Choose which ESM2 architecture your finetuned model is",
        action="store",
        default = 'esm2_t12_35M_UR50D',
)
    lang_gen.add_argument(
        "--temp",
        help="Choose sampling temperature.",
        action="store",
        default = '1',
)

    lang_gen.add_argument(
        "--ctrl_tag",
        help="Choose an Enzymatic Commision (EC) control tag for conditional protein generation based on the tag. You can find all ECs here https://www.brenda-enzymes.org/ecexplorer.php?browser=1",
        action="store",
)

    lang_gen.add_argument(
        "--seed_seq",
        help="Sequence to seed generation",
        default='M',
)
    lang_gen.add_argument(
        "--max_length",
        help="Max length of proteins generated",
        default=100,
        type=int
)
    lang_gen.add_argument(
        "--do_sample",
        help="Whether or not to use sampling for ProtGPT2 ; use greedy decoding otherwise",
        default=True,
        dest="do_sample",
)
    lang_gen.add_argument(
        "--top_k",
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering for ProtGPT2 or ESM2_Gibbs",
        default=950,
        dest="top_k",
        type=int
)
    lang_gen.add_argument(
        "--repetition_penalty",
        help="The parameter for repetition penalty for ProtGPT2. 1.0 means no penalty",
        default=1.2,
        dest="repetition_penalty",
)
    lang_gen.add_argument(
        "--num_return_sequences",
        help="Number of sequences for ProtGPT or ESM2_Gibbs to generate. Default is 5",
        default=5,
        dest="num_return_sequences",
        type=int,
)
    lang_gen.add_argument("--random_fill", 
        help="Randomly select positions to fill each iteration for Gibbs sampling with ESM2. If not called then fill the positions in order", 
        action="store_false",
        default = True,
        )
    lang_gen.add_argument("--num_positions", 
        help="Generate new AAs for this many positions each iteration for Gibbs sampling with ESM2. If 0, then generate for all target positions each round.", 
        action="store",
        default = 0,
        )
    
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
        default=None
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
    classify = subparsers.add_parser('classify', help='Classify proteins based on thermostability predicted through TemStaPro')

    classify.add_argument(
        "classifier",
        help="Predict thermostability using TemStaPro or choose custom to train/use your own XGBoost based binary classifier. Note for training a custom_binary, you need to submit roughly equal amounts of both binary classes as part of your query.",
        choices = ['TemStaPro', 'custom_binary']
)
    classify.add_argument(
        "query",
        help="Fasta file of sequences to score",
        action="store"
)
    classify.add_argument(
        "--key",
        help="String that allows for the unique identification of your binary classes from the input fasta headers. For example, --key positive_hits would group all sequences that have 'positive_hits' in the fasta header as one class and the rest as the other class",
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
        help="Select between 'esm2_t6_8M', 'esm2_t12_35M', 'esm2_t30_150M', 'esm2_t33_650M', 'esm2_t36_3B','esm2_t48_15B', or 'ProtT5-XL' for embedding your query proteins to then train your custom classifier",
        default = 'esm2_t12_35M',
        action="store"
)
    classify.add_argument(
        "--train_split",
        help="Choose your train-test percentage split for training and evaluating your custom classifier. For example, --train .6 would split your input sequences into two groups, one with 60%% of the sequences to train and the other with 40%% for evaluating",
        action="store",
)
    classify.add_argument(
        "--preTrained",
        help="Enter the path to your pre-trained XGBoost binary classifier that you've trained with TRILL.",
        action="store",
)
##############################################################################################################
    
    fold = subparsers.add_parser('fold', help='Predict 3D protein structures using ESMFold')

    fold.add_argument("query", 
        help="Input fasta file", 
        action="store"
        )
    fold.add_argument("--strategy", 
        help="Choose a specific strategy if you are running out of CUDA memory. You can also pass either 64, or 32 for model.trunk.set_chunk_size(x)", 
        action="store",
        default = None,
        )    
##############################################################################################################
    visualize = subparsers.add_parser('visualize', help='Reduce dimensionality of embeddings to 2D')

    visualize.add_argument("embeddings", 
        help="Embeddings to be visualized", 
        action="store"
        )
    
    visualize.add_argument("--method", 
        help="Method for reducing dimensions of embeddings. Default is PCA, but you can also choose UMAP or tSNE", 
        action="store",
        default="PCA"
        )
    visualize.add_argument("--group", 
        help="Grouping for color scheme of output scatterplot. Choose this option if the labels in your embedding csv are grouped by the last pattern separated by an underscore. For example, 'Protein1_group1', 'Protein2_group1', 'Protein3_group2'. By default, all points are treated as same group.", 
        action="store_true",
        default=False
        )
    
##############################################################################################################
    dock = subparsers.add_parser('dock', help='Dock protein to protein using DiffDock')

    dock.add_argument("protein", 
        help="Protein of interest to be docked with ligand", 
        action="store"
        )
    
    dock.add_argument("ligand", 
        help="Ligand to dock protein with", 
        action="store",
        )
    dock.add_argument("--save_visualisation", 
        help="Save a pdb file with all of the steps of the reverse diffusion.", 
        action="store_true",
        default=False
        )
    
    dock.add_argument("--samples_per_complex", 
        help="Number of samples to generate.", 
        type = int,
        action="store",
        default=10
        )
    
    dock.add_argument("--no_final_step_noise", 
        help="Use no noise in the final step of the reverse diffusion", 
        action="store_true",
        default=False
        )
    
    dock.add_argument("--inference_steps", 
        help="Number of denoising steps", 
        type=int,
        action="store",
        default=20
        )

    dock.add_argument("--actual_steps", 
        help="Number of denoising steps that are actually performed", 
        type=int,
        action="store",
        default=None
        )

    return parser
