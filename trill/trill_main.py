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
from tqdm import tqdm
import numpy as np
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from trill.utils.lightning_models import ESM, ProtGPT2, CustomWriter, ESM_Gibbs
from trill.utils.update_weights import weights_update
from transformers import AutoTokenizer, EsmForProteinFolding, set_seed
from pytorch_lightning.callbacks import ModelCheckpoint
# from trill.utils.strategy_tuner import tune_esm_inference, tune_esm_train
from trill.utils.protgpt2_utils import ProtGPT2_wrangle
from trill.utils.esm_utils import ESM_IF1_Wrangle, ESM_IF1, convert_outputs_to_pdb
from trill.utils.visualize import reduce_dims, viz

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
    generate = subparsers.add_parser('generate', help='Generate proteins using either ESM-IF1, ProtGPT2, ProteinMPNN or Gibbs sampling with a finetuned ESM2')
    generate.add_argument(
        "model",
        help="Choose between Inverse Folding model 'esm_if1_gvp4_t16_142M_UR50' to facilitate fixed backbone sequence design, ProteinMPNN or ProtGPT2.",
        choices = ['ESM-IF1','ProtGPT2', 'ProteinMPNN', 'ESM2_Gibbs']
)
    generate.add_argument(
        "--finetuned",
        help="Input path to your own finetuned ProtGPT2 or ESM2 model",
        action="store",
        default = False,
)
    generate.add_argument(
        "--esm2_arch",
        help="Choose which ESM2 architecture your finetuned model is",
        action="store",
        default = False,
)
    generate.add_argument(
        "--temp",
        help="Choose sampling temperature for ESM_IF1 or ProteinMPNN.",
        action="store",
        default = '1',
        dest="temp",
)
    
    generate.add_argument(
        "--genIters",
        help="Choose sampling iteration number for ESM_IF1.",
        action="store",
        default = 1,
        dest="genIters",
)

    generate.add_argument(
        "--seed_seq",
        help="Sequence to seed ProtGPT2 Generation",
        default='M',
        dest="seed_seq",
)
    generate.add_argument(
        "--max_length",
        help="Max length of proteins generated from ProtGPT2 or ProteinMPNN",
        default=1000,
        dest="max_length",
        type=int
)
    generate.add_argument(
        "--do_sample",
        help="Whether or not to use sampling for ProtGPT2 ; use greedy decoding otherwise",
        default=True,
        dest="do_sample",
)
    generate.add_argument(
        "--top_k",
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering for ProtGPT2 or ESM2_Gibbs",
        default=950,
        dest="top_k",
        type=int
)
    generate.add_argument(
        "--repetition_penalty",
        help="The parameter for repetition penalty for ProtGPT2. 1.0 means no penalty",
        default=1.2,
        dest="repetition_penalty",
)
    generate.add_argument(
        "--num_return_sequences",
        help="Number of sequences for ProtGPT2 or ProteinMPNN to generate. Default is 5",
        default=5,
        dest="num_return_sequences",
        type=int,
)

    generate.add_argument("--query", 
        help="Input pdb or cif file for inverse folding with ESM_IF1", 
        action="store"
        )
    



    generate.add_argument("--mpnn_model", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")
    generate.add_argument("--save_score", type=int, default=0, help="0 for False, 1 for True; save score=-log_prob to npy files")
    generate.add_argument("--save_probs", type=int, default=0, help="0 for False, 1 for True; save MPNN predicted probabilites per position")
    generate.add_argument("--score_only", type=int, default=0, help="0 for False, 1 for True; score input backbone-sequence pairs")
    generate.add_argument("--path_to_fasta", type=str, default="", help="score provided input sequence in a fasta format; e.g. GGGGGG/PPPPS/WWW for chains A, B, C sorted alphabetically and separated by /")
    generate.add_argument("--conditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)")    
    generate.add_argument("--conditional_probs_only_backbone", type=int, default=0, help="0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)") 
    generate.add_argument("--unconditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output unconditional probabilities p(s_i given backbone) in one forward pass")   
    generate.add_argument("--backbone_noise", type=float, default=0.00, help="Standard deviation of Gaussian noise to add to backbone atoms")
    generate.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    generate.add_argument("--pdb_path_chains", type=str, default='', help="Define which chains need to be designed for a single PDB ")
    generate.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    generate.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions")
    generate.add_argument("--omit_AAs", type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    generate.add_argument("--bias_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
    generate.add_argument("--bias_by_res_jsonl", default='', help="Path to dictionary with per position bias.") 
    generate.add_argument("--omit_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
    generate.add_argument("--pssm_jsonl", type=str, default='', help="Path to a dictionary with pssm")
    generate.add_argument("--pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
    generate.add_argument("--pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs")
    generate.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True")
    generate.add_argument("--pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True")
    generate.add_argument("--tied_positions_jsonl", type=str, default='', help="Path to a dictionary with tied positions")
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
                            pred_embeddings.append(tuple([sublist[0][0], sublist[0][1]]))
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
            model = ProtGPT2(int(args.lr))
            tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
            seq_dict_df = ProtGPT2_wrangle(data, tokenizer)
            dataloader = torch.utils.data.DataLoader(seq_dict_df, shuffle = False, batch_size = int(args.batch_size), num_workers=0)
            trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator='gpu', max_epochs=int(args.epochs), logger = logger, num_nodes = int(args.nodes), precision = 16, strategy = 'deepspeed_stage_2_offload')
            trainer.fit(model=model, train_dataloaders = dataloader)
            print(len_data)
            save_path = os.path.join(os.getcwd(), f"checkpoints/epoch={int(args.epochs) - 1}-step={len_data*int(args.epochs)}.ckpt")
            output_path = f"{args.name}_ProtGPT2_{args.epochs}.pt"
            try:
                convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
            except Exception as e:
                print(f'Exception {e} has occured on attempted save of your deepspeed trained model. If this has to do with CPU RAM, please try pytorch_lightning.utilities.deepspeedconvert_zero_checkpoint_to_fp32_state_dict(your_checkpoint.ckpt, full_model.pt')
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

    elif args.command == 'generate':
        if args.model == 'ProtGPT2':
            model = ProtGPT2(0.0001)
            if args.finetuned != False:
                model = model.load_from_checkpoint(args.finetuned, strict = False, lr = 0.0001)
            tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
            generated_output = []
            with open(f'{args.name}_ProtGPT2.fasta', 'w+') as fasta:
                for i in tqdm(range(int(args.num_return_sequences))):
                    generated_output = (model.generate(seed_seq=args.seed_seq, temperature = float(args.temp), max_length=int(args.max_length), do_sample = args.do_sample, top_k=int(args.top_k), repetition_penalty=float(args.repetition_penalty)))
                # gen_seq_df = pd.DataFrame(generated_output, columns=['Generated_Sequence'])
                    # for i, seq in enumerate(gen_seq_df['Generated_Sequence']):
                    fasta.write(f'>{args.name}_ProtGPT2_{i} \n')
                    fasta.write(f'{generated_output[0]}\n')
                    fasta.flush()
        elif args.model == 'ESM-IF1':
            if args.query == None:
                raise Exception('A PDB or CIF file is needed for generating new proteins with ESM-IF1')
            data = ESM_IF1_Wrangle(args.query)
            dataloader = torch.utils.data.DataLoader(data, pin_memory = True, batch_size=1, shuffle=False)
            sample_df = ESM_IF1(dataloader, genIters=int(args.genIters), temp = float(args.temp))
            sample_df.to_csv(f'{args.name}_IF1_gen.csv', index=False, header = ['Generated_Seq', 'Chain'])
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
        elif args.model == 'ESM2_Gibbs':
            model_import_name = f'esm.pretrained.{args.esm2_arch}()'
            if args.finetuned != False:
                model = ESM_Gibbs(eval(model_import_name), 0.0001, False, args.seed_seq, int(args.num_return_sequences), max_len=int(args.max_length), temp=float(args.temp), top_k = int(args.top_k))
                model = weights_update(model , checkpoint = torch.load(args.finetuned))
                tuned_name = args.finetuned.split('/')[-1]            
            else:
                model = ESM_Gibbs(eval(model_import_name), 0.0001, False, args.seed_seq, int(args.num_return_sequences), max_len=int(args.max_length), temp=float(args.temp), top_k = int(args.top_k))
                tuned_name = f'{args.esm2_arch}___'
            with open(f'{args.name}_{tuned_name[0:-3]}_Gibbs.fasta', 'w+') as fasta:
                for i in tqdm(range(int(args.num_return_sequences))):
                    gen_seq = model.predict()
                    fasta.write(f'>{args.name}_{tuned_name[0:-3]}_Gibbs_{i} \n')
                    fasta.write(f'{gen_seq}\n')
                    fasta.flush()


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
    generate = subparsers.add_parser('generate', help='Generate proteins using either ESM-IF1, ProtGPT2, ProteinMPNN or Gibbs sampling with a finetuned ESM2')
    generate.add_argument(
        "model",
        help="Choose between Inverse Folding model 'esm_if1_gvp4_t16_142M_UR50' to facilitate fixed backbone sequence design, ProteinMPNN or ProtGPT2.",
        choices = ['ESM-IF1','ProtGPT2', 'ProteinMPNN', 'ESM2_Gibbs']
)
    generate.add_argument(
        "--finetuned",
        help="Input path to your own finetuned ProtGPT2 or ESM2 model",
        action="store",
        default = False,
)
    generate.add_argument(
        "--esm2_arch",
        help="Choose which ESM2 architecture your finetuned model is",
        action="store",
        default = False,
)
    generate.add_argument(
        "--temp",
        help="Choose sampling temperature for ESM_IF1 or ProteinMPNN.",
        action="store",
        default = '1',
        dest="temp",
)
    
    generate.add_argument(
        "--genIters",
        help="Choose sampling iteration number for ESM_IF1.",
        action="store",
        default = 1,
        dest="genIters",
)

    generate.add_argument(
        "--seed_seq",
        help="Sequence to seed ProtGPT2 Generation",
        default='M',
        dest="seed_seq",
)
    generate.add_argument(
        "--max_length",
        help="Max length of proteins generated from ProtGPT2 or ProteinMPNN",
        default=1000,
        dest="max_length",
)
    generate.add_argument(
        "--do_sample",
        help="Whether or not to use sampling for ProtGPT2 ; use greedy decoding otherwise",
        default=True,
        dest="do_sample",
)
    generate.add_argument(
        "--top_k",
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering for ProtGPT2 or ESM2_Gibbs",
        default=950,
        dest="top_k",
)
    generate.add_argument(
        "--repetition_penalty",
        help="The parameter for repetition penalty for ProtGPT2. 1.0 means no penalty",
        default=1.2,
        dest="repetition_penalty",
)
    generate.add_argument(
        "--num_return_sequences",
        help="Number of sequences for ProtGPT2 or ProteinMPNN to generate. Default is 5",
        default=5,
        dest="num_return_sequences",
)

    generate.add_argument("--query", 
        help="Input pdb or cif file for inverse folding with ESM_IF1", 
        action="store"
        )
    



    generate.add_argument("--mpnn_model", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")
    generate.add_argument("--save_score", type=int, default=0, help="0 for False, 1 for True; save score=-log_prob to npy files")
    generate.add_argument("--save_probs", type=int, default=0, help="0 for False, 1 for True; save MPNN predicted probabilites per position")
    generate.add_argument("--score_only", type=int, default=0, help="0 for False, 1 for True; score input backbone-sequence pairs")
    generate.add_argument("--path_to_fasta", type=str, default="", help="score provided input sequence in a fasta format; e.g. GGGGGG/PPPPS/WWW for chains A, B, C sorted alphabetically and separated by /")
    generate.add_argument("--conditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)")    
    generate.add_argument("--conditional_probs_only_backbone", type=int, default=0, help="0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)") 
    generate.add_argument("--unconditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output unconditional probabilities p(s_i given backbone) in one forward pass")   
    generate.add_argument("--backbone_noise", type=float, default=0.00, help="Standard deviation of Gaussian noise to add to backbone atoms")
    generate.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    generate.add_argument("--pdb_path_chains", type=str, default='', help="Define which chains need to be designed for a single PDB ")
    generate.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    generate.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions")
    generate.add_argument("--omit_AAs", type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    generate.add_argument("--bias_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
    generate.add_argument("--bias_by_res_jsonl", default='', help="Path to dictionary with per position bias.") 
    generate.add_argument("--omit_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
    generate.add_argument("--pssm_jsonl", type=str, default='', help="Path to a dictionary with pssm")
    generate.add_argument("--pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
    generate.add_argument("--pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs")
    generate.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True")
    generate.add_argument("--pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True")
    generate.add_argument("--tied_positions_jsonl", type=str, default='', help="Path to a dictionary with tied positions")
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
    




    

