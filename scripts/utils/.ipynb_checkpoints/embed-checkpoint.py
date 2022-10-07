import pandas as pd
import numpy as np
import glob
import re
import io
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import esm
from argparse import Namespace
from esm.constants import proteinseq_toks
from esm.modules import TransformerLayer # noqa
from esm.model import ProteinBertModel
from tqdm import tqdm
from esm.pretrained import esm1b_t33_650M_UR50S, esm1_t12_85M_UR50S
from esm.data import FastaBatchedDataset


def generate_embedding_transformer_t12(model,batch_converter,dat,name,seq_col):
	if torch.cuda.is_available():
		device = 'cuda'
	else:
		device = 'cpu'
	if device == 'cuda':
		model.cuda()
	sequence_embeddings = []
	model.eval()
	for epoch in tqdm(range(dat.shape[0])):
		data = [(dat.iloc[epoch, 1], dat.iloc[epoch, seq_col])]
		_, _, batch_tokens = batch_converter(data)
		with torch.no_grad():
			if device == 'cuda':
				results = model(tokens = batch_tokens.to('cuda'), repr_layers=[12])
			else:
				results = model(tokens = batch_tokens.to('cpu'), repr_layers=[12])
			token_embeddings = results["representations"][12]
			seq = dat.iloc[epoch,seq_col]
			sequence_embeddings.append(token_embeddings[0, 1:len(seq) + 1].mean(0).cpu().detach().numpy())

	sequence_embeddings = np.array(sequence_embeddings)
	np.save('../data/' + name + ".npy", sequence_embeddings)

def embed(tuned_model, query, database, name):
    
	model_t12, alphabet = esm1_t12_85M_UR50S()

	if torch.cuda.is_available():
		model_t12 = model_t12.cuda('cuda')
	else:
		model_t12 = model_t12.cpu()
        
	if tuned_model != 'N' or '':
		model_t12.load_state_dict(torch.load(tuned_model))
		q = pd.read_csv(query)
		db = pd.read_csv(database)
		master_db = pd.concat([q, db], axis = 0)

	else:
		master_db = pd.read_csv(query)

	batch_converter = alphabet.get_batch_converter()

	generate_embedding_transformer_t12(model_t12,batch_converter,master_db,name,seq_col = 1)

	return True