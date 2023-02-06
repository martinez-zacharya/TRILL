import pytorch_lightning as pl
import torch
import argparse
import esm
import time
import gc
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
from tqdm import tqdm
sys.path.insert(0, 'utils')
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from trill.utils.lightning_models import ESM, ProtGPT2, CustomWriter
from trill.utils.update_weights import weights_update
from transformers import AutoTokenizer, EsmForProteinFolding
from pytorch_lightning.callbacks import ModelCheckpoint
# from trill.utils.strategy_tuner import tune_esm_inference, tune_esm_train
from trill.utils.protgpt2_utils import ProtGPT2_wrangle
from trill.utils.esm_utils import ESM_IF1_Wrangle, ESM_IF1, convert_outputs_to_pdb
from trill.utils.visualize import reduce_dims, viz
from pyfiglet import Figlet
import bokeh
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):

    torch.set_float32_matmul_precision('medium')
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
    generate = subparsers.add_parser('generate', help='Generate proteins using either ESM-IF1 or ProtGPT2')
    generate.add_argument(
        "model",
        help="Choose between Inverse Folding model 'esm_if1_gvp4_t16_142M_UR50' to facilitate fixed backbone sequence design or ProtGPT2.",
        choices = ['ESM-IF1','ProtGPT2']
)
    generate.add_argument(
        "--finetuned_protgpt2",
        help="Input path to your own finetuned ProtGPT2 model",
        action="store",
        default = False,
)
    generate.add_argument(
        "--temp",
        help="Choose sampling temperature for ESM_IF1.",
        action="store",
        default = 1.,
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
        help="Max length of proteins generated from ProtGPT2",
        default=333,
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
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering for ProtGPT2",
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
        help="Number of sequences for ProtGPT2 to generate. Default is 5",
        default=5,
        dest="num_return_sequences",
)

    generate.add_argument("--query", 
        help="Input pdb or cif file for inverse folding with ESM_IF1", 
        action="store"
        )
##############################################################################################################
    fold = subparsers.add_parser('fold', help='Predict 3D protein structures using ESMFold')

    fold.add_argument("query", 
        help="Input fasta file", 
        action="store"
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


    

    

    args = parser.parse_args()
    start = time.time()
    # if args.query == None and args.gen == False:
    #     raise ValueError('An input file is needed when not using --gen')

    pl.seed_everything(123)
    
    
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
            if args.finetuned_protgpt2 == True:
                model = model.load_from_checkpoint(args.preTrained_model, strict = False, lr = 0.0001)
            tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
            generated_output = model.generate(seed_seq=args.seed_seq, max_length=int(args.max_length), do_sample = args.do_sample, top_k=int(args.top_k), repetition_penalty=float(args.repetition_penalty), num_return_sequences=int(args.num_return_sequences))
            gen_seq_df = pd.DataFrame(generated_output, columns=['Generated_Sequence'])
            gen_seq_df.to_csv(f'{args.name}_generated_sequences.csv', index = False)
        elif args.model == 'ESM-IF1':
            if args.query == None:
                raise Exception('A PDB or CIF file is needed for generating new proteins with ESM-IF1')
            data = ESM_IF1_Wrangle(args.query)
            dataloader = torch.utils.data.DataLoader(data, pin_memory = True, batch_size=1, shuffle=False)
            sample_df = ESM_IF1(dataloader, genIters=int(args.genIters), temp = args.temp)
            sample_df.to_csv(f'{args.name}_IF1_gen.csv', index=False, header = ['Generated_Seq', 'Chain'])

    elif args.command == 'fold':
        print(f'Initializing esmfold_v1...')
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", device_map="auto")
        model.esm = model.esm.half()
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
                except Exception as e:
                    torch.cuda.empty_cache()
                    print(f'Protein too long to fold for current hardware: {prot_len} amino acids long')
                    print(e)
                    del tokenized_input

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
    generate = subparsers.add_parser('generate', help='Generate proteins using either ESM-IF1 or ProtGPT2')
    generate.add_argument(
        "model",
        help="Choose between Inverse Folding model 'esm_if1_gvp4_t16_142M_UR50' to facilitate fixed backbone sequence design or ProtGPT2.",
        choices = ['ESM-IF1','ProtGPT2']
)
    generate.add_argument(
        "--finetuned_protgpt2",
        help="Input path to your own finetuned ProtGPT2 model",
        action="store",
        default = False,
)
    generate.add_argument(
        "--temp",
        help="Choose sampling temperature for ESM_IF1.",
        action="store",
        default = 1.,
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
        help="Max length of proteins generated from ProtGPT2",
        default=333,
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
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering for ProtGPT2",
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
        help="Number of sequences for ProtGPT2 to generate. Default is 5",
        default=5,
        dest="num_return_sequences",
)

    generate.add_argument("--query", 
        help="Input pdb or cif file for inverse folding with ESM_IF1", 
        action="store"
        )
##############################################################################################################
    fold = subparsers.add_parser('fold', help='Predict 3D protein structures using ESMFold')
    fold.add_argument("query", 
        help="Input fasta file", 
        action="store"
        )
##############################################################################################################
##############################################################################################################
    visualize = subparsers.add_parser('visualize', help='Reduce dimensionality of embeddings to 2D')

    visualize.add_argument("embeddings", 
        help="Embeddings in a csv to be visualized with last column as label", 
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


    
    return parser
    




    

