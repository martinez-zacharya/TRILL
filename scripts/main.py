import sys
import argparse
import pandas as pd
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

	args = parser.parse_args()

	name = args.name
	query = args.query
	database = args.database
	lr = int(args.lr)
	epochs = int(args.epochs)

	FineTuneQueryValidation(query)
	FineTuneDatabaseValidation(database)

	finetune('query_df.csv', name, lr, epochs)

	model_name = 'esm_t12_85M_UR50S_' + name + '.pt'
	embed(model_name, 'query_df.csv', 'database_df.csv', name)

	master_db = pd.concat([pd.read_csv('query_df_labeled.csv'), pd.read_csv('database_df_labeled.csv')], axis = 0).reset_index(drop = True)

	tsnedf = tsne(name, master_db)

	scatter_viz(tsnedf)