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


# def generate_embedding_transformer_t12(model,batch_converter,dat,name,seq_col):
# 	if torch.cuda.is_available():
# 		device = 'cuda'
# 	else:
# 		device = 'cpu'
# 	if device == 'cuda':
# 		model.cuda()
# 	sequence_embeddings = []
# 	model.eval()
# 	for epoch in tqdm(range(dat.shape[0])):
# 		data = [(dat.iloc[epoch, 1], dat.iloc[epoch, seq_col])]
# 		_, _, batch_tokens = batch_converter(data)
# 		with torch.no_grad():
# 			if device == 'cuda':
# 				results = model(tokens = batch_tokens.to('cuda'), repr_layers=[12])
# 			else:
# 				results = model(tokens = batch_tokens.to('cpu'), repr_layers=[12])
# 			token_embeddings = results["representations"][12]
# 			seq = dat.iloc[epoch,seq_col]
# 			sequence_embeddings.append(token_embeddings[0, 1:len(seq) + 1].mean(0).cpu().detach().numpy())

# 	sequence_embeddings = np.array(sequence_embeddings)
# 	np.save('../data/' + name + ".npy", sequence_embeddings)

# def embed(tuned_model, query, database, name):
    
# 	model_t12, alphabet = esm1_t12_85M_UR50S()

# 	if torch.cuda.is_available():
# 		model_t12 = model_t12.cuda('cuda')
# 	else:
# 		model_t12 = model_t12.cpu()
        
# 	if tuned_model != 'N' or '':
# 		model_t12.load_state_dict(torch.load(tuned_model))
# 		q = pd.read_csv(query)
# 		db = pd.read_csv(database)
# 		master_db = pd.concat([q, db], axis = 0)

# 	else:
# 		master_db = pd.read_csv(query)

# 	batch_converter = alphabet.get_batch_converter()

# 	generate_embedding_transformer_t12(model_t12,batch_converter,master_db,name,seq_col = 1)

# 	return True

def embed_setup(gpu, args):
	torch.cuda.empty_cache()
	rank = gpu
	torch.manual_seed(0)

	dist.init_process_group(                                   
	backend='nccl',                                         
	init_method='env://',
	timeout = timedelta(seconds = 300),                                 
	world_size=int(args.GPUs),                              
	rank=rank                                               
	)
	model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
 
	fsdp_params = dict(
		mixed_precision=False,
		flatten_parameters=True,
		state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
		cpu_offload=True,  # enable cpu offloading
	)

	with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
		model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
		if args.preTrained_model == True:
			model.load_state_dict(torch.load('subsamp1000_VP1s_20eps_esm2t30.pt'))
		batch_converter = alphabet.get_batch_converter()
		model.eval()

		# Wrap each layer in FSDP separately
		for name, child in model.named_children():
			if name == "layers":
				for layer_name, layer in child.named_children():
					wrapped_layer = wrap(layer)
					setattr(child, layer_name, wrapped_layer)
		model = wrap(model)
  
	gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

	local_rank = rank - gpus_per_node * (rank // gpus_per_node)

	torch.cuda.set_device(local_rank)



	dataset = FastaBatchedDataset.from_file(args.query)

	train_sampler = torch.utils.data.distributed.DistributedSampler(
	dat,
	num_replicas=int(args.GPUs),
	rank=rank
	)
	data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_size = args.batch_size)

	print(f"Read {fasta_file} with {len(dataset)} sequences")
	
	#define the output layer
	assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in [-1])
	repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in [-1]]

	#now we can obtain the representations by inputin the tokens into the model and taken the last layer's embedding
	# result_reps = []
	result_reps = [()]
	with torch.no_grad():
		for batch_idx, (labels, strs, toks) in enumerate(data_loader):
			print(f"Processing {batch_idx + 1} of {len(data_loader)} batches ({toks.size(0)} sequences)")
			if torch.cuda.is_available():
				toks = toks.to(f"cuda:{local_rank}")
			# The model is trained on truncated sequences and passing longer ones in at
			# infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
			
			# toks = toks[:, :1022]
		
			out = model(toks, repr_layers=repr_layers, return_contacts=False)

			#   logits = out["logits"].to(device="cpu")
			representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
			
			#convert these to a numpy array because that is more useful
			rep_numpy = representations[-1].cpu().detach().numpy()
			for i in range(len(rep_numpy)):
				result_reps.append((rep_numpy[i].mean(0), labels[i]))
			# GPUtil.showUtilization()

	newdf = pd.DataFrame(result_reps, columns = ['Embeddings', 'Label'])
	newdf = newdf.drop(index=newdf.index[0], axis=0)
	finaldf = newdf['Embeddings'].apply(pd.Series)
	finaldf['Label'] = newdf['Label']
	finaldf.to_csv(f'{args.name}_{args.query[0:-6]}_esm2t30.csv', index = False)
	return result_reps