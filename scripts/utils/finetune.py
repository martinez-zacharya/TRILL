import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import esm
import pandas as pd
import time
import sys
import numpy as np
from tqdm import tqdm
sys.path.insert(0, '../esm')
from argparse import Namespace
from esm.constants import proteinseq_toks
from esm.pretrained import esm1b_t33_650M_UR50S, esm1_t12_85M_UR50S
from esm.modules import TransformerLayer
from esm.model import ProteinBertModel
from esm.data import FastaBatchedDataset
from ych_util import prepare_mlm_mask
from torch.utils.data import Dataset

def finetune(infile, tuned_name, lr, epochs):
# 	dat = pd.read_csv(infile, names = ['Protein', 'Seq'])
	dat = FastaBatchedDataset.from_file('some_proteins.fasta')
# 	model, alphabet = esm1b_t33_650M_UR50S()
	model_, alphabet = esm1_t12_85M_UR50S()

	if torch.cuda.is_available():
		device = "cuda"
		model_ = model_.cuda()
	else:
		device = "cpu"

	dat_loader = torch.utils.data.DataLoader(dat, batch_size = 1, num_workers=1, shuffle = True, collate_fn=alphabet.get_batch_converter())

	model_ = nn.DataParallel(model_)
	model_.train()

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model_.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

	for j in tqdm(range(epochs)):
		for i, (labels, seq, toks) in enumerate(dat_loader):
			if len(seq) > 1024:
				seq = seq[:1022]
				toks = toks[:1022]
			else:
				seq = seq
				toks = toks
			true_aa,target_ind,masked_batch_tokens = prepare_mlm_mask(alphabet,toks)
			optimizer.zero_grad()
			if device == 'cuda':
				masked_batch_tokens = masked_batch_tokens.to(device = 'cuda', non_blocking=True)
				results = model_(tokens = masked_batch_tokens, repr_layers=[12])   

			else:
				results = model_(masked_batch_tokens.to('cpu'), repr_layers=[12])
                
			pred = results["logits"].squeeze(0)[target_ind,:]   
			target = true_aa.squeeze(0)
			loss = criterion(pred.cuda(),target.cuda())
			loss.backward()
			optimizer.step()
			torch.save(model_.state_dict(), f"esm1_t12_85M_UR50S_{tuned_name}.pt")
		print(j, loss)

	return True