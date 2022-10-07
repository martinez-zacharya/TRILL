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
sys.path.insert(0, 'utils')
from lightning_model import LitModel
from update_weights import weights_update



os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def main():
    start = time.time()
    
    
    pl.seed_everything(123)
    model_import_name = f'esm.pretrained.{args.model}()'
    model = LitModel(eval(model_import_name), float(args.lr))
    data = esm.data.FastaBatchedDataset.from_file(args.query)
    dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = int(args.batch_size), num_workers=0, collate_fn=model.alphabet.get_batch_converter())
    if args.logger == True:
        logger = TensorBoardLogger("logs")
    else:
        logger = False
    
    
    if args.noTrain == True:
        trainer = pl.Trainer(devices=int(args.GPUs), strategy = args.strategy, accelerator='gpu', logger=logger, num_nodes=int(args.nodes))
        trainer.predict(model, dataloader)
        newdf = pd.DataFrame(model.reps, columns = ['Embeddings', 'Label'])
        newdf = newdf.drop(index=newdf.index[0], axis=0)
        finaldf = newdf['Embeddings'].apply(pd.Series)
        finaldf['Label'] = newdf['Label']
        finaldf.to_csv(f'{args.name}_{args.model}.csv', index = False)
    
    elif args.preTrained_model != False:
        model = weights_update(model = LitModel(eval(model_import_name), float(args.lr)), checkpoint = torch.load('/home/zacharymartinez/DistantHomologyDetection/scripts/test_esm2_t12_35M_UR50D_20.pt'))
        trainer = pl.Trainer(devices=int(args.GPUs), strategy = args.strategy, accelerator='gpu', logger=logger, num_nodes=int(args.nodes))
        trainer.predict(model, dataloader)
        newdf = pd.DataFrame(model.reps, columns = ['Embeddings', 'Label'])
        newdf = newdf.drop(index=newdf.index[0], axis=0)
        finaldf = newdf['Embeddings'].apply(pd.Series)
        finaldf['Label'] = newdf['Label']
        finaldf.to_csv(f'{args.name}_{args.model}.csv', index = False)
        
    elif args.blast == True:
        trainer = pl.Trainer(devices=int(args.GPUs), accelerator='gpu', strategy = args.strategy, max_epochs=int(args.epochs), logger=logger, num_nodes=int(args.nodes))        
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
    
    else:
        trainer = pl.Trainer(devices=int(args.GPUs), accelerator='gpu', strategy = args.strategy, max_epochs=int(args.epochs), logger=logger, num_nodes=int(args.nodes), precision = 16, amp_backend='native')        
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
        help="Input database to search through", 
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
        help="Enables BLAST mode",
        action="store_true",
        default = False,
        dest="blast",
)
    
    parser.add_argument(
        "--model",
        help="Change ESM model. Default is esm2_t12_35M_UR50D",
        action="store",
        default = 'esm2_t12_35M_UR50D',
        dest="model",
)

    parser.add_argument(
        "--strategy",
        help="Change training strategy. Default is None",
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
        
    


    args = parser.parse_args()

    
    main()