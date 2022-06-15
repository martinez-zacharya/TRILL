import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import esm
import pandas as pd
from datetime import datetime
import sys
import socket
import os
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
sys.path.insert(0, 'esm')
import argparse
from esm.constants import proteinseq_toks
from esm.pretrained import esm1b_t33_650M_UR50S, esm1_t12_85M_UR50S
from esm.modules import TransformerLayer
from esm.model import ProteinBertModel
from esm.data import FastaBatchedDataset
from mlm import prepare_mlm_mask
from torch.utils.data import Dataset
import subprocess
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

def finetune(gpu, fasta, tuned_name, lr, epochs, world_size):
	rank = int(os.environ['SLURM_JOB_NUM_NODES']) * len(os.environ['SLURM_JOB_GPUS']) + int(gpu)                          
	torch.manual_seed(0)
	hostname = os.environ['SLURM_JOB_NODELIST']
	ip_add = subprocess.run(["nslookup", hostname], stdout = subprocess.PIPE)
	ip = ip_add.stdout.decode("utf-8")
	ip = ip.split('\t')
	ip = ip.pop(-1)
	ip = ip.split('\n')
	# ip = ip.pop(0)
	# ip = ip.split(' ')
	# ip = ip[1]
	print(ip)

	#os.environ['MASTER_ADDR'] = ip
	os.environ['MASTER_PORT'] = '12345'
	dist.init_process_group(                                   
    backend='nccl',                                         
   	init_method='env://',                                   
    world_size=world_size,                              
    rank=rank                                               
    )  
    
	dat = FastaBatchedDataset.from_file(fasta)
	model_, alphabet = esm1_t12_85M_UR50S()

	if torch.cuda.is_available():
		torch.cuda.set_device(gpu)
		model_ = model_.cuda(gpu)
	else:
		device = "cpu"
		return("Don't run this on a CPU")

	train_sampler = torch.utils.data.distributed.DistributedSampler(
		dat,
		num_replicas=world_size,
		rank=rank
	)

	dat_loader = torch.utils.data.DataLoader(dat, batch_size = 5, num_workers=0, pin_memory=True, sampler = train_sampler, collate_fn=alphabet.get_batch_converter())

	ddp_model = DDP(model_, device_ids=[rank], find_unused_parameters=True)
	ddp_model.train()

	criterion = nn.CrossEntropyLoss(reduction='none').cuda('cuda')
	optimizer = torch.optim.Adam(ddp_model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    
	start = datetime.now()

	for j in tqdm(range(epochs)):
		for i, (labels, seq, toks) in enumerate(dat_loader):

			if len(seq) > 1024:
				seq = seq[:1022]
				toks = toks[:1022]
			else:
				seq = seq
				toks = toks
			true_aa, masked_batch_tokens = prepare_mlm_mask(alphabet, toks)
			optimizer.zero_grad()
			if torch.cuda.is_available():
				masked_batch_tokens = masked_batch_tokens.to(gpu, non_blocking=True)
				results = ddp_model(tokens = masked_batch_tokens, repr_layers=[12])   

			else:
				results = model_(masked_batch_tokens.to('cpu'), repr_layers=[12])
                
			pred = results['logits']
			loss = criterion(pred.permute(0,2,1).to('cuda'),true_aa.to('cuda')).mean(dim=1)
			optimizer.zero_grad()
			loss.sum().backward()
			optimizer.step()
            
		print(j, loss)
	if gpu == 0:
		print("Training complete in: " + str(datetime.now() - start))
		torch.save(ddp_model.module.state_dict(), f"esm1_t12_85M_UR50S_{tuned_name}.pt")
	return True
