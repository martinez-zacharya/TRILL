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
sys.path.insert(0, 'esm')
import argparse
from esm.constants import proteinseq_toks
from esm.pretrained import esm1b_t33_650M_UR50S, esm1_t12_85M_UR50S
from esm.modules import TransformerLayer
from esm.model import ProteinBertModel
from esm.data import FastaBatchedDataset
from mlm import prepare_mlm_mask
from torch.distributed.fsdp.fully_sharded_data_parallel import (
FullyShardedDataParallel as FSDP,
CPUOffload,
BackwardPrefetch,
)
# from torch.distributed.fsdp.wrap import (
# default_auto_wrap_policy,
# enable_wrap,
# wrap,
# )
import torch.distributed as dist
from torch.utils.data import Dataset
import datetime
import subprocess
from fairscale.experimental.nn.offload import OffloadModel
from fairscale.experimental.tooling.layer_memory_tracker import LayerwiseMemoryTracker
from GPUtil import showUtilization as gpu_usage
from torch.cuda.amp import autocast
import torch.multiprocessing as mp
#from mlflow import log_metric, log_param, log_artifacts
from torch.nn.parallel import DistributedDataParallel as DDP
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

def finetune(gpu, args):
	torch.cuda.empty_cache()
	rank = int(os.environ["SLURM_NODEID"])  * 4 + gpu
	torch.cuda.set_device(rank)
	torch.manual_seed(0)
	dist.init_process_group(                                   
    backend='nccl',                                         
   	init_method='env://',
   	timeout = datetime.timedelta(seconds = 300),                                 
    world_size=int(args.GPUs),                              
    rank=rank                                               
    )  
	
    
	dat = FastaBatchedDataset.from_file(args.query)
	model_, alphabet = esm1_t12_85M_UR50S()

	if torch.cuda.is_available():
		torch.cuda.set_device(gpu)
		model_ = model_.to(gpu)
	else:
		device = "cpu"
		return("Don't run this on a CPU")

	train_sampler = torch.utils.data.distributed.DistributedSampler(
		dat,
		num_replicas=int(args.GPUs),
		rank=rank
	)

	dat_loader = torch.utils.data.DataLoader(dat, batch_size = int(args.batch_size), num_workers=0, pin_memory = True, sampler = train_sampler, collate_fn=alphabet.get_batch_converter())


	criterion = nn.CrossEntropyLoss(reduction='none').to(gpu)

	ddp_model = FSDP(
        model_,
        cpu_offload=CPUOffload(offload_params=True),
)
	optimizer = torch.optim.Adam(ddp_model.parameters(), lr = int(args.lr))

	ddp_model.train() 
    
    
# 	tracker = LayerwiseMemoryTracker()
# 	tracker.monitor(ddp_model)
    

# 	start = datetime.now()
	accumulation_steps = 1
	optimizer.zero_grad()
	for j in tqdm(range(int(args.epochs))):
		for i, (labels, seq, toks) in enumerate(dat_loader):
			del labels, seq
            
			torch.cuda.empty_cache()

			if toks.shape[1] > 1024:
				toks = torch.narrow(toks, 1, 0, 1023)

			true_aa, masked_batch_tokens = prepare_mlm_mask(alphabet, toks)
			optimizer.zero_grad()
			if torch.cuda.is_available():
				masked_batch_tokens = masked_batch_tokens.to(gpu, non_blocking=True)
				results = ddp_model(tokens = masked_batch_tokens, repr_layers=[12])
	
			else:
				results = model_(masked_batch_tokens.to('cpu'), repr_layers=[12])
                
			pred = results['logits']
			loss = criterion(pred.permute(0,2,1).to(gpu),true_aa.to(gpu))
			loss = loss.mean()
			loss = loss.sum()

# 			tracker.show_plots()

			loss.backward()
			#log_metric('CrossEntropyLoss', loss)
            
# 			if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
# 				optimizer.step() # Now we can do an optimizer step
# 				optimizer.zero_grad()
			optimizer.step()                            # Now we can do an optimizer step
			optimizer.zero_grad()
		print(j, loss)
        
	states = ddp_model.state_dict()
	if gpu == 0:

# 		print("Training complete in: " + str(datetime.now() - start))
		torch.save(
            states,
            f"esm1_t12_85M_UR50S_{args.name}.pt"
        )
    
	dist.destroy_process_group()
# 	tracker.stop()

	return True
