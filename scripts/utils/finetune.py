import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import esm
import pandas as pd
import time
from tqdm import tqdm
from argparse import Namespace
from esm.constants import proteinseq_toks
from esm.modules import TransformerLayer, PositionalEmbedding
from esm.model import ProteinBertModel
from ych_util import prepare_mlm_mask

def finetune(infile, tuned_name, lr, epochs):
	dat = pd.read_csv(infile)
	dat = dat.sample(frac = 1)
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
	model_state = {prs(arg[0]): arg[1] for arg in model_data["model"].items()}
	model = esm.ProteinBertModel(
		Namespace(**model_args), len(alphabet), padding_idx=alphabet.padding_idx
    )

	model.load_state_dict(model_state)
	if device == 'gpu':
		model.cuda()

	model = nn.DataParallel(model)
	model.train()

	batch_converter = alphabet.get_batch_converter()
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
	start_time = time.time()

	for j in tqdm(range(epochs)):
		dat = dat.sample(frac = 1)
		for i in range(dat.shape[0]):
			if len(dat.iloc[i,1])>1024:
				seq = dat.iloc[i,1][:1023]
			else:
				seq = dat.iloc[i,1]
			lab = dat.iloc[i,0]
			data = [(lab, seq)]
			batch_labels, batch_strs, batch_tokens = batch_converter(data)
			true_aa,target_ind,masked_batch_tokens = prepare_mlm_mask(alphabet,batch_tokens)
			optimizer.zero_grad()
			if device == 'gpu':
				results = model(masked_batch_tokens.to('cuda'), repr_layers=[34])   
			else:
				results = model(masked_batch_tokens.to('cpu'), repr_layers=[34])   

			pred = results["logits"].squeeze(0)[target_ind,:]   
			target = true_aa.squeeze(0)
			loss = criterion(pred.cpu(),target)
			loss.backward()
			optimizer.step()

			torch.save(model.state_dict(), f"esm_t12_85M_UR50S_{tuned_name}.pt")

	return True