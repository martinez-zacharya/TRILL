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
from esm.modules import TransformerLayer, PositionalEmbedding  # noqa
from esm.model import ProteinBertModel
from tqdm import tqdm


def generate_embedding_transformer_t12(model,batch_converter,dat,name,seq_col):
	if torch.cuda.is_available():
		device = 'gpu'
	else:
		device = 'cpu'
	if device == 'gpu':
		model.cuda()
	sequence_embeddings = []
	model = nn.DataParallel(model)
	for epoch in tqdm(range(dat.shape[0])):
		data = [(dat.iloc[epoch, 1], dat.iloc[epoch, seq_col])]
		_, _, batch_tokens = batch_converter(data)
		with torch.no_grad():
			if device == 'gpu':
				results = model(batch_tokens.to('cuda'), repr_layers=[12])
			else:
				results = model(batch_tokens.to('cpu'), repr_layers=[12])
			token_embeddings = results["representations"][12]
			seq = dat.iloc[epoch,seq_col]
			sequence_embeddings.append(token_embeddings[0, 1:len(seq) + 1].mean(0).cpu().detach().numpy())

	sequence_embeddings = np.array(sequence_embeddings)
	np.save('../data/' + name + ".npy", sequence_embeddings)

def embed(tuned_model, query, database, name):
	torch.save(tuned_model.state_dict(), 'pre-trained.pth')
	alphabet = esm.Alphabet.from_dict(proteinseq_toks)
	if torch.cuda.is_available():
		device = 'gpu'
	else:
		device = 'cpu'
	url = "https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t12_85M_UR50S.pt"
	if device == 'gpu':
		model_data = torch.hub.load_state_dict_from_url(url, progress=False)
	else:
		model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location=torch.device('cpu'))

	pra = lambda s: ''.join(s.split('decoder_')[1:] if 'decoder' in s else s)
	prs = lambda s: ''.join(s.split('decoder.')[1:] if 'decoder' in s else s)
	model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}

	if tuned_model != 'N' or '':
		model_t12 = torch.load(tuned_model)
	else:
		model_t12 = esm.ProteinBertModel(Namespace(**model_args), len(alphabet), padding_idx=alphabet.padding_idx)
		model_state_12 = {prs(arg[0]): arg[1] for arg in model_data["model"].items()}
		model_t12.load_state_dict(model_state_12)


	# model_t12 = esm.ProteinBertModel(Namespace(**model_args), len(alphabet), padding_idx=alphabet.padding_idx)

	q = pd.read_csv(query)
	db = pd.read_csv(database)

	master_db = pd.concat([q, db], axis = 0)
	batch_converter = alphabet.get_batch_converter()

	generate_embedding_transformer_t12(model_t12,batch_converter,master_db,name,seq_col = 1)

	return True