import sys
import argparse
import os
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

if __name__ == "__main__":

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
		"database", 
		help="Input database to search through", 
		action="store"
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
		help="Number of epochs for fine-tuning transformer. Default is 300",
		action="store",
		default=300,
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

	args = parser.parse_args()

	name = args.name
	query = args.query
	database = args.database
	lr = int(args.lr)
	epochs = int(args.epochs)
	noTrain_flag = args.noTrain
	preTrained_model = args.preTrained_model

	world_size = len(os.environ['SLURM_JOB_GPUS']) * int(os.environ['SLURM_JOB_NUM_NODES'])

	# This is for when you just want to embed the raw sequences
	if noTrain_flag == True:
		FineTuneQueryValidation(name, query)
		FineTuneDatabaseValidation(name, database)

		embed('N', f'{name}_query_df.csv', f'{name}_database_df.csv', name)

		master_db = pd.concat([pd.read_csv(f'{name}_query_df_labeled.csv'), pd.read_csv(f'{name}_database_df_labeled.csv')], axis = 0).reset_index(drop = True)

		tsnedf = tsne(name, master_db)

		scatter_viz(tsnedf)
	elif preTrained_model != False:
		FineTuneQueryValidation(name, query)
		FineTuneDatabaseValidation(name, database)

		embed(preTrained_model, f'{name}_query_df.csv', f'{name}_database_df.csv', name)

		master_db = pd.concat([pd.read_csv(f'{name}_query_df_labeled.csv'), pd.read_csv(f'{name}_database_df_labeled.csv')], axis = 0).reset_index(drop = True)

		tsnedf = tsne(name, master_db)

		scatter_viz(tsnedf)
	else:
		FineTuneQueryValidation(name, query)
		FineTuneDatabaseValidation(name, database)

		mp.spawn(finetune, nprocs = len(os.environ['SLURM_JOB_GPUS']), args = (query,name, lr, epochs, world_size))

		model_name = 'esm_t12_85M_UR50S_' + name + '.pt'
		embed(model_name, f'{name}_query_df.csv', f'{name}_database_df.csv', name)

		# master_db = pd.concat([pd.read_csv(f'{name}_query_df_labeled.csv'), pd.read_csv(f'{name}_database_df_labeled.csv')], axis = 0).reset_index(drop = True)

		# tsnedf = tsne(name, master_db)

		# scatter_viz(tsnedf)
