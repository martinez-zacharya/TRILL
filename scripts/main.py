import pytorch_lightning as pl
import torch
import argparse
import esm
import time
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from tqdm import tqdm
sys.path.insert(0, 'utils')
from lightning_model import LitModel, coordDataset
from update_weights import weights_update
from esm.inverse_folding.util import load_structure, extract_coords_from_structure
from esm.inverse_folding.multichain_util import extract_coords_from_complex, sample_sequence_in_complex

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def main():
    start = time.time()
    
    
    pl.seed_everything(123)
    model_import_name = f'esm.pretrained.{args.model}()'
    if args.if1 == True:
        pass
    else:
        model = LitModel(eval(model_import_name), float(args.lr))
    if args.query.endswith(('.pdb', '.cif')) == True:
        structures = load_structure(args.query)
        data = extract_coords_from_complex(structures)
        data = coordDataset([data])
        dataloader = torch.utils.data.DataLoader(data, pin_memory = True, batch_size=1, shuffle=False)
        
    elif args.query.endswith(('.fasta', '.faa', '.fa')):
        data = esm.data.FastaBatchedDataset(args.query)
        dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = int(args.batch_size), num_workers=0, collate_fn=model.alphabet.get_batch_converter())

    else:
        return (f'Input query file - {args.query} is not a valid file format.\
            File needs to be either protein fasta (.fa, .fasta, .faa) or atomic coordinates (.pdb, .cif)')
    
    if args.logger == True:
        logger = TensorBoardLogger("logs")
    else:
        logger = False
    
    
    if args.noTrain == True:
        trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), strategy = args.strategy, accelerator='gpu', logger=logger, num_nodes=int(args.nodes))
        trainer.predict(model, dataloader)
        newdf = pd.DataFrame(model.reps, columns = ['Embeddings', 'Label'])
        newdf = newdf.drop(index=newdf.index[0], axis=0)
        finaldf = newdf['Embeddings'].apply(pd.Series)
        finaldf['Label'] = newdf['Label']
        finaldf.to_csv(f'{args.name}_{args.model}.csv', index = False)
    
    elif args.preTrained_model != False:
        model = weights_update(model = LitModel(eval(model_import_name), float(args.lr)), checkpoint = torch.load('/home/zacharymartinez/DistantHomologyDetection/scripts/test_esm2_t12_35M_UR50D_20.pt'))
        trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), strategy = args.strategy, accelerator='gpu', logger=logger, num_nodes=int(args.nodes))
        trainer.predict(model, dataloader)
        newdf = pd.DataFrame(model.reps, columns = ['Embeddings', 'Label'])
        newdf = newdf.drop(index=newdf.index[0], axis=0)
        finaldf = newdf['Embeddings'].apply(pd.Series)
        finaldf['Label'] = newdf['Label']
        finaldf.to_csv(f'{args.name}_{args.model}.csv', index = False)
        
    elif args.blast == True:
        trainer = pl.Trainer(devices=int(args.GPUs), precision = 16, amp_backend='native', accelerator='gpu', strategy = args.strategy, max_epochs=int(args.epochs), logger=logger, num_nodes=int(args.nodes), enable_checkpointing=False)        
        trainer.fit(model=model, train_dataloaders=dataloader)
        trainer.save_checkpoint(f"{args.name}_{args.model}_{args.epochs}.pt")
        blastdb = esm.data.FastaBatchedDataset.from_file(args.database)
        blastdb_loader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = int(args.batch_size), num_workers=0, collate_fn=model.alphabet.get_batch_converter())
        trainer.predict(model, dataloader)
        trainer.predict(model, blastdb_loader)
        newdf = pd.DataFrame(model.reps, columns = ['Embeddings', 'Label'])
        newdf = newdf.drop(index=newdf.index[0], axis=0)
        finaldf = newdf['Embeddings'].apply(pd.Series)
        finaldf['Label'] = newdf['Label']
        finaldf.to_csv(f'{args.name}_{args.model}.csv', index = False)       
    
    elif args.if1 == True:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval()
        sampled_seqs = [()]
        for batch in data:
            coords, native_seq = batch
            chains = list(coords.keys())
            loop_chain = tqdm(chains)
            loop_chain.set_description('Chains')
            for chain in loop_chain:
                loop_gen_iters = tqdm(range(int(args.genIters)))
                loop_gen_iters.set_description('Generative Iterations')
                for i in loop_gen_iters:
                    sampled_seq = sample_sequence_in_complex(model, coords, chain, temperature=args.temp)
                    sampled_seqs.append(tuple([sampled_seq, chain]))
        sample_df = pd.DataFrame(sampled_seqs)
        sample_df = sample_df.iloc[1: , :]
        sample_df.to_csv(f'{args.name}_IF1_gen.csv', index=False, header = ['Generated_Seq', 'Chain'])      
        
        
        
    else:
        trainer = pl.Trainer(devices=int(args.GPUs), accelerator='gpu', strategy = args.strategy, max_epochs=int(args.epochs), logger=logger, num_nodes=int(args.nodes), precision = 16, amp_backend='native', enable_checkpointing=False)        
        trainer.fit(model=model, train_dataloaders=dataloader)
        trainer.save_checkpoint(f"{args.name}_{args.model}_{args.epochs}.pt")
        
    
    end = time.time()
    print("Finished!")
    print(f"Time elapsed: {end-start} seconds")
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "name",
        help = "Name of run",
        action = "store"
        )

    parser.add_argument("query", 
        help="Input fasta file for queries", 
        action="store"
        )

    parser.add_argument(
        "--database", 
        help="Input database to embed with --blast mode", 
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
        help="Learning rate for adam optimizer. Default is 0.0001",
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
        default = 5,
        dest="batch_size",
)

    parser.add_argument(
        "--blast",
        help="Enables 'BLAST' mode. --database argument is required",
        action="store_true",
        default = False,
        dest="blast",
)
    
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
    
    parser.add_argument(
        "--chain",
        help="Choose which chain to predict while using --if1 mode. Default is A",
        action="store",
        default = 'A',
        dest="chain",
)
    
    parser.add_argument(
        "--temp",
        help="Choose sampling temperature.",
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
        
    


    args = parser.parse_args()

    
    main()