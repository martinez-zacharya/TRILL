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
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
import yaml
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from trill.utils.lightning_models import ESM, ProtGPT2, CustomWriter, ESM_Gibbs, ProtT5
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

    subparsers = parser.add_subparsers(dest='command')

    embed = subparsers.add_parser('embed', help='Embed proteins of interest')

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
        "--preTrained_model",
        help="Input path to your own pre-trained ESM model",
        action="store",
        default = False,
        dest="preTrained_model",
)
    embed.add_argument(
        "--model",
        help="Change model. Default is esm2_t12_35M_UR50D. You can choose from a list of ESM2 models which can be found at https://github.com/facebookresearch/esm",
        action="store",
        default = 'esm2_t12_35M_UR50D',
        dest="model",
)
##############################################################################################################

    finetune = subparsers.add_parser('finetune', help='Fine-tune models')

    finetune.add_argument("query", 
        help="Input fasta file", 
        action="store"
)
    finetune.add_argument("--epochs", 
        help="Number of epochs for fine-tuning. Default is 20", 
        action="store",
        default=20,
        dest="epochs",
        )
    finetune.add_argument(
        "--lr",
        help="Learning rate for optimizer. Default is 0.0001",
        action="store",
        default=0.0001,
        dest="lr",
)
    finetune.add_argument(
        "--model",
        help="Change model. Default is esm2_t12_35M_UR50D. You can choose either ProtGPT2 or various ESM2. List of ESM2 models can be found at https://github.com/facebookresearch/esm",
        action="store",
        default = 'esm2_t12_35M_UR50D',
        dest="model",
)
    finetune.add_argument(
        "--LEGGO",
        help="deepspeed_stage_3_offload.",
        action="store_true",
        default=False,
        dest="LEGGO",
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
        help="Choose between Inverse Folding model 'esm_if1_gvp4_t16_142M_UR50' to facilitate fixed backbone sequence design, ProteinMPNN or ProtGPT2.",
        choices = ['ESM2','ProtGPT2']
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
        help="Choose sampling temperature for ESM_IF1 or ProteinMPNN.",
        action="store",
        default = '1',
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
        help="Change RFDiffusion model.",
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
    
    # diffuse_gen.add_argument("--RFDiffusion_yaml", 
    #     help="Specify RFDiffusion params using a yaml file. Easiest option for complicated runs",
    #     action="store",
    #     default = None
    #     )

##############################################################################################################
    classify = subparsers.add_parser('classify', help='Generate proteins using large language models including ProtGPT2 and ESM2')

    classify.add_argument(
        "classifier",
        help="Predict thermostability using TemStaPro",
        choices = ['TemStaPro']
)
    classify.add_argument(
        "query",
        help="Fasta file of sequences to score with TemStaPro",
        action="store"
)
    classify.add_argument(
        "--save_emb",
        help="Save csv of ProtT5 embeddings",
        action="store_true",
        default=False
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


    

    

    args = parser.parse_args()
    start = time.time()
    # if args.query == None and args.gen == False:
    #     raise ValueError('An input file is needed when not using --gen')

    pl.seed_everything(int(args.RNG_seed))
    set_seed(int(args.RNG_seed))
    
    
    torch.backends.cuda.matmul.allow_tf32 = True
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
        else:
            model_import_name = f'esm.pretrained.{args.model}()'
            model = ESM(eval(model_import_name), 0.0001, False)
            data = esm.data.FastaBatchedDataset.from_file(args.query)
            dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = int(args.batch_size), num_workers=0, collate_fn=model.alphabet.get_batch_converter())
            pred_writer = CustomWriter(output_dir=".", write_interval="epoch")
            trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], devices=int(args.GPUs), accelerator='gpu', logger=logger, num_nodes=int(args.nodes))

            if args.preTrained_model == False:
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
                model = weights_update(model = ESM(eval(model_import_name), 0.0001, False), checkpoint = torch.load(args.preTrained_model))
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
            model = ProtGPT2(int(args.lr), args.strategy)
            tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
            seq_dict_df = ProtGPT2_wrangle(data, tokenizer)
            dataloader = torch.utils.data.DataLoader(seq_dict_df, shuffle = False, batch_size = int(args.batch_size), num_workers=0)
            trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator='gpu', max_epochs=int(args.epochs), logger = logger, num_nodes = int(args.nodes), precision = 16, strategy = args.strategy)
            trainer.fit(model=model, train_dataloaders = dataloader)
            if 'deepspeed' in str(args.strategy):
                save_path = os.path.join(os.getcwd(), f"checkpoints/epoch={int(args.epochs) - 1}-step={len_data*int(args.epochs)}.ckpt")
                output_path = f"{args.name}_ProtGPT2_{args.epochs}.pt"
                try:
                    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
                except Exception as e:
                    print(f'Exception {e} has occured on attempted save of your deepspeed trained model. If this has to do with CPU RAM, please try pytorch_lightning.utilities.deepspeedconvert_zero_checkpoint_to_fp32_state_dict(your_checkpoint.ckpt, full_model.pt')
            else:
                trainer.save_checkpoint(f"{args.name}_{args.model}_{args.epochs}.pt")
        else:
            model_import_name = f'esm.pretrained.{args.model}()'
            model = ESM(eval(model_import_name), float(args.lr), args.LEGGO)
            dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = int(args.batch_size), num_workers=0, collate_fn=model.alphabet.get_batch_converter())
            if args.LEGGO:
                trainer = pl.Trainer(devices=int(args.GPUs), profiler = profiler,accelerator='gpu',max_epochs=int(args.epochs),logger=logger, num_nodes=int(args.nodes), precision = 16, strategy=DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True))
                trainer.fit(model=model, train_dataloaders=dataloader)
                save_path = os.path.join(os.getcwd(), f"checkpoints/epoch={int(args.epochs) - 1}-step={len_data*int(args.epochs)}.ckpt")
                output_path = f"{args.name}_esm2_{args.epochs}.pt"
                try:
                    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
                except Exception as e:
                    print(f'Exception {e} has occured on attempted save of your deepspeed trained model. If this has to do with CPU RAM, please try pytorch_lightning.utilities.deepspeedconvert_zero_checkpoint_to_fp32_state_dict(your_checkpoint.ckpt, full_model.pt')       
            elif args.strategy == 'deepspeed_stage_3' or args.strategy == 'deepspeed_stage_3_offload' or args.strategy == 'deepspeed_stage_2' or args.strategy == 'deepspeed_stage_2_offload':
                save_path = os.path.join(os.getcwd(), f"checkpoints/epoch={int(args.epochs) - 1}-step={len_data*int(args.epochs)}.ckpt")
                output_path = f"{args.name}_esm2_{args.epochs}.pt"
                trainer = pl.Trainer(devices=int(args.GPUs), profiler = profiler, accelerator='gpu', strategy = args.strategy, max_epochs=int(args.epochs), logger=logger, num_nodes=int(args.nodes), precision = 16,  enable_checkpointing=False)        
                trainer.fit(model=model, train_dataloaders=dataloader)
                trainer.save_checkpoint(f"{args.name}_{args.model}_{args.epochs}.pt")
                try:
                    convert_zero_checkpoint_to_fp32_state_dict(f"{args.name}_{args.model}_{args.epochs}.pt", output_path)
                except Exception as e:
                    print(f'Exception {e} has occured on attempted save of your deepspeed trained model. If this has to do with CPU RAM, please try pytorch_lightning.utilities.deepspeedconvert_zero_checkpoint_to_fp32_state_dict(your_checkpoint.ckpt, full_model.pt')       
            else:
                trainer = pl.Trainer(devices=int(args.GPUs), profiler = profiler, accelerator='gpu', strategy = args.strategy, max_epochs=int(args.epochs), logger=logger, num_nodes=int(args.nodes), precision = 16, amp_backend='native', enable_checkpointing=False)        
                trainer.fit(model=model, train_dataloaders=dataloader)
                trainer.save_checkpoint(f"{args.name}_{args.model}_{args.epochs}.pt")

    elif args.command == 'inv_fold_gen':
        if args.model == 'ESM-IF1':
            if args.query == None:
                raise Exception('A PDB or CIF file is needed for generating new proteins with ESM-IF1')
            data = ESM_IF1_Wrangle(args.query)
            dataloader = torch.utils.data.DataLoader(data, pin_memory = True, batch_size=1, shuffle=False)
            sample_df, native_seq_df = ESM_IF1(dataloader, genIters=int(args.num_return_sequences), temp = float(args.temp))
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
                subprocess.run(['pip', 'install', mpnn_git_root])
            else:
                sys.path.insert(0, 'ProteinMPNN/')
            from mpnnrun import run_mpnn
            print('ProteinMPNN generation starting...')
            run_mpnn(args)
        
    elif args.command == 'lang_gen':
        if args.model == 'ProtGPT2':
            model = ProtGPT2(0.0001, None)
            if args.finetuned != False:
                model = model.load_from_checkpoint(args.finetuned, strict = False, lr = 0.0001, strat = None)
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
                    model = ESM_Gibbs(eval(model_import_name))
                    if args.finetuned != False:
                        model = weights_update(model = ESM_Gibbs(eval(model_import_name)), checkpoint = torch.load(args.finetuned))
                        tuned_name = args.finetuned.split('/')[-1] 
                    for i in range(args.num_return_sequences):
                        out = model.generate(args.seed_seq, mask=True, n_samples = 1, max_len = args.max_length, in_order = args.random_fill, num_positions=int(args.num_positions), temperature=float(args.temp))
                        out = ''.join(out)
                        fasta.write(f'>{args.name}_{tuned_name[0:-3]}_Gibbs_{i} \n')
                        fasta.write(f'{out}\n')
                        fasta.flush()           
                else:
                    model = ESM_Gibbs(eval(model_import_name))
                    tuned_name = f'{args.esm2_arch}___'
                    for i in range(args.num_return_sequences):
                        out = model.generate(args.seed_seq, mask=True, n_samples = 1, max_len = args.max_length, in_order = args.random_fill, num_positions=int(args.num_positions), temperature=float(args.temp))
                        out = ''.join(out)
                        fasta.write(f'>{args.name}_{tuned_name[0:-3]}_Gibbs_{i} \n')
                        fasta.write(f'{out}\n')
                        fasta.flush()  
    elif args.command == 'diff_gen':
        os.environ['HYDRA_FULL_ERROR'] = '1'
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

        
        run_rfdiff((f'{rfdiff_git_root}/config/inference/base.yaml'), args)

    elif args.command == 'fold':
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

        model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1', low_cpu_mem_usage=True, torch_dtype='auto')
        # model.esm = model.esm.half()
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
                            break
                        else:
                            print(e)
                            pass

        pdb_list = [convert_outputs_to_pdb(output) for output in outputs]
        protein_identifiers = fold_df.Entry.tolist()
        for identifier, pdb in zip(protein_identifiers, pdb_list):
            with open(f"{identifier}.pdb", "w") as f:
                f.write("".join(pdb))

    elif args.command == 'dock':

        # if args.ligand.endswith(".pdb") or args.ligand.endswith(".PDB"):
        #     mol = Chem.MolFromPDBFile(args.ligand)
        #     sdf_file = f"{args.ligand.split('.')[0]}.sdf"
        #     writer = Chem.SDWriter(sdf_file)
        #     writer.write(mol)
        #     writer.close()
        #     args.ligand = sdf_file

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
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        model = ProtT5()
        dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = 1, num_workers=0)
        trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), accelerator='gpu', logger=logger, num_nodes=int(args.nodes))
        trainer.predict(model, dataloader)
        if args.save_emb:
            emb_4export = []
            for rep in model.reps:
                intermed = []
                for i in range(len(rep[0])):
                    intermed.append(rep[0][i].cpu().numpy())
                intermed.append(rep[1][0])
                emb_4export.append(intermed)
            emb_df = pd.DataFrame(emb_4export)
            emb_df.to_csv(f'{args.name}_prot_t5_xl_half_uniref50-enc_embeddings.csv', index=False)
        if not os.path.exists('TemStaPro_models/'):
            temstapro_models = Repo.clone_from('https://github.com/martinez-zacharya/TemStaPro_models', 'TemStaPro_models/')
            temstapro_models_root = temstapro_models.git.rev_parse("--show-toplevel")
        else:
            temstapro_models = Repo('TemStaPro_models')
            temstapro_models_root = temstapro_models.git.rev_parse("--show-toplevel")
        dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = 1, num_workers=0)
        THRESHOLDS = ["40", "45", "50", "55", "60", "65"]
        SEEDS = ["41", "42", "43", "44", "45"]
        emb_loader = torch.utils.data.DataLoader(model.reps, shuffle = False, batch_size = 1, num_workers = 0)
        inferences = {}
        for thresh in THRESHOLDS:
            threshold_inferences = {}
            for seed in SEEDS:
                clf = MLP_C2H2(1024, 512, 256)
                clf.load_state_dict(torch.load(f'{temstapro_models_root}/mean_major_imbal-{thresh}_s{seed}.pt'))
                clf.eval()
                clf.to('cuda')
                threshold_inferences[seed] = inference_epoch(clf, emb_loader, device='cuda')
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

    subparsers = parser.add_subparsers(dest='command')

    embed = subparsers.add_parser('embed', help='Embed proteins of interest')

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
        "--preTrained_model",
        help="Input path to your own pre-trained ESM model",
        action="store",
        default = False,
        dest="preTrained_model",
)
    embed.add_argument(
        "--model",
        help="Change model. Default is esm2_t12_35M_UR50D. You can choose from a list of ESM2 models which can be found at https://github.com/facebookresearch/esm",
        action="store",
        default = 'esm2_t12_35M_UR50D',
        dest="model",
)
##############################################################################################################

    finetune = subparsers.add_parser('finetune', help='Fine-tune models')

    finetune.add_argument("query", 
        help="Input fasta file", 
        action="store"
)
    finetune.add_argument("--epochs", 
        help="Number of epochs for fine-tuning. Default is 20", 
        action="store",
        default=20,
        dest="epochs",
        )
    finetune.add_argument(
        "--lr",
        help="Learning rate for optimizer. Default is 0.0001",
        action="store",
        default=0.0001,
        dest="lr",
)
    finetune.add_argument(
        "--model",
        help="Change model. Default is esm2_t12_35M_UR50D. You can choose either ProtGPT2 or various ESM2. List of ESM2 models can be found at https://github.com/facebookresearch/esm",
        action="store",
        default = 'esm2_t12_35M_UR50D',
        dest="model",
)
    finetune.add_argument(
        "--LEGGO",
        help="deepspeed_stage_3_offload.",
        action="store_true",
        default=False,
        dest="LEGGO",
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
        help="Choose between Inverse Folding model 'esm_if1_gvp4_t16_142M_UR50' to facilitate fixed backbone sequence design, ProteinMPNN or ProtGPT2.",
        choices = ['ESM-IF1','ProtGPT2']
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
        help="Choose sampling temperature for ESM_IF1 or ProteinMPNN.",
        action="store",
        default = '1',
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
        help="Change RFDiffusion model.",
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
    
    # diffuse_gen.add_argument("--RFDiffusion_yaml", 
    #     help="Specify RFDiffusion params using a yaml file. Easiest option for complicated runs",
    #     action="store",
    #     default = None
    #     )

    
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

    return parser
