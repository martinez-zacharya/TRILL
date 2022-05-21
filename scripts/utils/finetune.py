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

	dat_loader = torch.utils.data.DataLoader(dat, batch_size = 100, shuffle = True, num_workers=64, pin_memory = True)
	model = nn.DataParallel(model)
	model.train()

	batch_converter = alphabet.get_batch_converter()
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
	start_time = time.time()

	for j in tqdm(range(epochs)):
		# dat = dat.sample(frac = 1)
		for i, data in enumerate(dat_loader):
			label, seq = data
			if len(seq)>1024:
				seq = seq[:1023]
			else:
				seq = seq
			processed_data = [(label, seq)]
			batch_labels, batch_strs, batch_tokens = batch_converter(processed_data)
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
		print(j, loss)

	return True