import sys
import argparse
import os
import subprocess
import datetime
import torch.distributed as dist
import pandas as pd
import torch.multiprocessing as mp
sys.path.insert(0, 'utils')
from WranglingData import (
    FineTuneQueryValidation,
    FineTuneDatabaseValidation
    )
from finetune import (
    finetune
    )
from embed import (
    embed,
    generate_embedding_transformer_t12
    )
from visualize import (
    tsne,
    scatter_viz
    )

def main():

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
        help="Input total number of GPUs",
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
        help="Number of epochs for fine-tuning transformer. Default is 100",
        action="store",
        default=100,
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
        help="Change batch-size number for fine-tuning",
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



    args = parser.parse_args()

    name = args.name
    query = args.query
    database = args.database
    lr = int(args.lr)
    epochs = int(args.epochs)
    noTrain_flag = args.noTrain
    preTrained_model = args.preTrained_model
    
    os.environ['MASTER_PORT'] = '8888'
    # This is for when you just want to embed the raw sequences
    if noTrain_flag == True:
        FineTuneQueryValidation(name, query)

        embed('N', f'{name}_query_df.csv', f'{name}_database_df.csv', name)

#         master_db = pd.concat([pd.read_csv(f'{name}_query_df_labeled.csv'), pd.read_csv(f'{name}_database_df_labeled.csv')], axis = 0).reset_index(drop = True)

#         tsnedf = tsne(name, master_db)

#         scatter_viz(tsnedf)
    elif preTrained_model != False:
        FineTuneQueryValidation(name, query)
        FineTuneDatabaseValidation(name, database)

        embed(preTrained_model, f'{name}_query_df.csv', f'{name}_database_df.csv', name)

#         master_db = pd.concat([pd.read_csv(f'{name}_query_df_labeled.csv'), pd.read_csv(f'{name}_database_df_labeled.csv')], axis = 0).reset_index(drop = True)

#         tsnedf = tsne(name, master_db)

#         scatter_viz(tsnedf)
    elif args.blast == True:
        FineTuneQueryValidation(name, query)
        FineTuneDatabaseValidation(name, database)

    
        mp.spawn(finetune, nprocs = 4, args = (args,), join = True)


        model_name = 'esm1_t12_85M_UR50S_' + name + '.pt'
        embed(model_name, f'{name}_query_df.csv', f'{name}_database_df.csv', name)

#         master_db = pd.concat([pd.read_csv(f'{name}_query_df_labeled.csv'), pd.read_csv(f'{name}_database_df_labeled.csv')], axis = 0).reset_index(drop = True)

#         tsnedf = tsne(name, master_db)

#         scatter_viz(tsnedf)

    else:
        FineTuneQueryValidation(name, query)
        if int(args.GPUs) >= 4:
                nprocs = 4
        else:
            nprocs = int(args.GPUs)
        
    
        mp.spawn(finetune, nprocs = nprocs, args = (args,), join = True)


        model_name = 'esm1_t12_85M_UR50S_' + name + '.pt'
        

    print("Finished!")
if __name__ == '__main__':
    main()
