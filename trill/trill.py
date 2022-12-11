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
import git
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
from tqdm import tqdm
sys.path.insert(0, 'utils')
from trill.utils.lightning_models import ESM, ProtGPT2
from trill.utils.update_weights import weights_update
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from esm.inverse_folding.util import load_structure, extract_coords_from_structure
from esm.inverse_folding.multichain_util import extract_coords_from_complex, sample_sequence_in_complex
from pytorch_lightning.callbacks import ModelCheckpoint
from trill.utils.protgpt2_utils import ProtGPT2_wrangle
from trill.utils.esm_utils import ESM_IF1_Wrangle, coordDataset, clean_embeddings, ESM_IF1
from pyfiglet import Figlet
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):

    f = Figlet(font="graffiti")
    print(f.renderText("TRILL"))
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "name",
        help = "Name of run",
        action = "store"
        )

    parser.add_argument("--query", 
        help="Input fasta file", 
        action="store"
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
        "--lr",
        help="Learning rate for optimizer. Default is 0.0001",
        action="store",
        default=0.0001,
        dest="lr",
)

    parser.add_argument(
        "--epochs",
        help="Number of epochs for fine-tuning transformer. Default is 20",
        action="store",
        default=20,
        dest="epochs",
)
    parser.add_argument(
        "--noTrain",
        help="Skips the fine-tuning and embeds the query and database sequences with the raw model",
         action="store_true",
        default = False,
        dest="noTrain",
)
    parser.add_argument(
        "--preTrained_model",
        help="Input path to your own pre-trained ESM model",
        action="store",
        default = False,
        dest="preTrained_model",
)
    
    parser.add_argument(
        "--batch_size",
        help="Change batch-size number for fine-tuning. Default is 5",
        action="store",
        default = 1,
        dest="batch_size",
)

#     parser.add_argument(
#         "--blast",
#         help="Enables 'BLAST' mode. --database argument is required",
#         action="store_true",
#         default = False,
#         dest="blast",
# )
    
    parser.add_argument(
        "--model",
        help="Change ESM model. Default is esm2_t12_35M_UR50D. List of models can be found at https://github.com/facebookresearch/esm",
        action="store",
        default = 'esm2_t12_35M_UR50D',
        dest="model",
)

    parser.add_argument(
        "--strategy",
        help="Change training strategy. Default is None. List of strategies can be found at https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html",
        action="store",
        default = None,
        dest="strategy",
)
    parser.add_argument(
        "--logger",
        help="Enable Tensorboard logger. Default is None",
        action="store",
        default = False,
        dest="logger",
)
    
    parser.add_argument(
        "--if1",
        help="Utilize Inverse Folding model 'esm_if1_gvp4_t16_142M_UR50' to facilitate fixed backbone sequence design. Basically converts protein structure to possible sequences.",
        action="store_true",
        default = False,
        dest="if1",
)
    
#     parser.add_argument(
#         "--chain",
#         help="Choose which chain to predict while using --if1 mode. Default is A",
#         action="store",
#         default = 'A',
#         dest="chain",
# )
    
    parser.add_argument(
        "--temp",
        help="Choose sampling temperature for --if1 mode.",
        action="store",
        default = 1.,
        dest="temp",
)
    
    parser.add_argument(
        "--genIters",
        help="Choose sampling iteration number for IF1.",
        action="store",
        default = 1,
        dest="genIters",
)
    parser.add_argument(
        "--LEGGO",
        help="deepspeed_stage_3_offload.",
        action="store_true",
        default=False,
        dest="LEGGO",
)

    parser.add_argument(
        "--profiler",
        help="Utilize PyTorchProfiler",
        action="store_true",
        default=False,
        dest="profiler",
)
    
    parser.add_argument(
        "--protgpt2",
        help="Utilize ProtGPT2. Can either fine-tune or generate sequences",
        action="store_true",
        default=False,
        dest="protgpt2",
)
    
    parser.add_argument(
        "--gen",
        help="Generate protein sequences using ProtGPT2. Can either use base model or user-submitted fine-tuned model",
        action="store_true",
        default=False,
        dest="gen",
)
    parser.add_argument(
        "--seed_seq",
        help="Sequence to seed ProtGPT2 Generation",
        default='M',
        dest="seed_seq",
)
    parser.add_argument(
        "--max_length",
        help="Max length of proteins generated from ProtGPT2",
        default=333,
        dest="max_length",
)
    parser.add_argument(
        "--do_sample",
        help="Whether or not to use sampling ; use greedy decoding otherwise",
        default=True,
        dest="do_sample",
)
    parser.add_argument(
        "--top_k",
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering",
        default=950,
        dest="top_k",
)
    parser.add_argument(
        "--repetition_penalty",
        help="The parameter for repetition penalty. 1.0 means no penalty",
        default=1.2,
        dest="repetition_penalty",
)
    parser.add_argument(
        "--num_return_sequences",
        help="Number of sequences for ProtGPT2 to generate",
        default=5,
        dest="num_return_sequences",
)

    args = parser.parse_args()
    start = time.time()
    if args.query == None and args.gen == False:
        raise ValueError('An input file is needed when not using --gen')

    pl.seed_everything(123)
    
    if args.logger == True:
        logger = TensorBoardLogger("logs")
    else:
        logger = False
    if args.profiler:
        profiler = PyTorchProfiler(filename='test-logs')
    else:
        profiler = None
        
    model_import_name = f'esm.pretrained.{args.model}()'
    if args.if1 == True:
        pass
    # elif args.esmfold == True:
    #     model = ESMFold()
    elif args.protgpt2 == True and args.preTrained_model == False:
        model = ProtGPT2(int(args.lr))
        tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
    elif args.protgpt2 == True and args.preTrained_model != False:
        model = ProtGPT2(int(args.lr))
        model = model.load_from_checkpoint(args.preTrained_model, strict = False)
        tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
    else:
        model = ESM(eval(model_import_name), float(args.lr), args.LEGGO)
        
    if args.query != None:
        if args.query.endswith(('.pdb', '.cif')):
            data = ESM_IF1_Wrangle(args.query)
            dataloader = torch.utils.data.DataLoader(data, pin_memory = True, batch_size=1, shuffle=False)
        
        elif args.query.endswith(('.fasta', '.faa', '.fa')):
            data = esm.data.FastaBatchedDataset.from_file(args.query)
            if args.protgpt2 == True:
                seq_dict_df = ProtGPT2_wrangle(data, tokenizer)
                dataloader = torch.utils.data.DataLoader(seq_dict_df, shuffle = False, batch_size = int(args.batch_size), num_workers=0)
            else:
            # if args.esmfold == True:
            #     dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = int(args.batch_size), num_workers=0)
            # else:
                dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = int(args.batch_size), num_workers=0, collate_fn=model.alphabet.get_batch_converter())

        else:
            return (f'Input query file - {args.query} is not a valid file format.\
            File needs to be either protein fasta (.fa, .fasta, .faa) or atomic coordinates (.pdb, .cif)')

        
    if args.noTrain == True:
        trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), strategy = args.strategy, accelerator='gpu', logger=logger, num_nodes=int(args.nodes))
        trainer.predict(model, dataloader)
        embeddings = clean_embeddings(model.reps)
        embeddings.to_csv(f'{args.name}_{args.model}.csv', index = False)
    
    elif args.preTrained_model != False and args.protgpt2 == False:
        model = weights_update(model = ESM(eval(model_import_name), float(args.lr)), checkpoint = torch.load(args.preTrained_model))
        trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), strategy = args.strategy, accelerator='gpu', logger=logger, num_nodes=int(args.nodes))
        trainer.predict(model, dataloader)
        embeddings = clean_embeddings(model.reps)
        embeddings.to_csv(f'{args.name}_{args.model}.csv', index = False)
        
    elif args.if1 == True:
        sample_df = ESM_IF1(dataloader, genIters=int(args.genIters), temp = args.temp)
        sample_df.to_csv(f'{args.name}_IF1_gen.csv', index=False, header = ['Generated_Seq', 'Chain'])      
        
    elif args.protgpt2 == True:
        if args.gen == True:
            generated_output = model.generate(seed_seq=args.seed_seq, max_length=int(args.max_length), do_sample = args.do_sample, top_k=int(args.top_k), repetition_penalty=float(args.repetition_penalty), num_return_sequences=int(args.num_return_sequences))
            gen_seq_df = pd.DataFrame(generated_output, columns=['Generated_Sequence'])
            gen_seq_df.to_csv(f'{args.name}_generated_sequences.csv', index = False)
        else:
            trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator='gpu', max_epochs=int(args.epochs), logger = logger, num_nodes = int(args.nodes), precision = 16, amp_backend='native', strategy = 'deepspeed_stage_3')
            trainer.fit(model=model, train_dataloaders = dataloader)
            save_path = os.path.join(os.getcwd(), f"lightning_logs/version_0/checkpoints/epoch={args.epochs}-step={len(seq_dict_df)}.ckpt")
            output_path = f"{args.name}_ProtGPT2_{args.epochs}.pt"
            convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
            # trainer.save_checkpoint(f"{args.name}_{args.epochs}.pt")
    # elif args.esmfold == True:
    #     trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs),  precision = 16, amp_backend='native', strategy = DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True), accelerator='gpu', logger=logger, num_nodes=int(args.nodes))
    #     trainer.predict(model, dataloader)
    #     print(model.preds)
    #     pdb_df = pd.DataFrame(model.preds)
    #     print(pdb_df)
    else:
        if args.LEGGO:
            trainer = pl.Trainer(devices=int(args.GPUs), profiler = profiler,accelerator='gpu',max_epochs=int(args.epochs),logger=logger, num_nodes=int(args.nodes), precision = 16, amp_backend='native', strategy=DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True))
        else:
            trainer = pl.Trainer(devices=int(args.GPUs), profiler = profiler, accelerator='gpu', strategy = args.strategy, max_epochs=int(args.epochs), logger=logger, num_nodes=int(args.nodes), precision = 16, amp_backend='native', enable_checkpointing=False)        
        trainer.fit(model=model, train_dataloaders=dataloader)
        trainer.save_checkpoint(f"{args.name}_{args.model}_{args.epochs}.pt")
        
    
    end = time.time()
    print("Finished!")
    print(f"Time elapsed: {end-start} seconds")
 

def cli(args=None):
    if not args:
        args = sys.argv[1:]    
    main(args)
if __name__ == '__main__':
    print("this shouldn't show up...")

    
#     parser.add_argument(
#         "--esmfold",
#         help="Predict protein structures in bulk using ESMFold",
#         action="store_true",
#         default=False,
#         dest="esmfold",
# ) 
    
    
        
    




    
   # main()
