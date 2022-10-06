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
from esm.data import FastaBatchedDataset
from mlm import prepare_mlm_mask
from mlm_martin import maskInputs
# from torch.distributed.fsdp.fully_sharded_data_parallel import (
# FullyShardedDataParallel as FSDP,
# CPUOffload,
# BackwardPrefetch,
# )
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
import torch.distributed as dist
from torch.utils.data import Dataset
from datetime import datetime, timedelta
import subprocess
# from fairscale.experimental.tooling.layer_memory_tracker import LayerwiseMemoryTracker
# from GPUtil import showUtilization as gpu_usage
from torch.cuda.amp import autocast
import torch.multiprocessing as mp
#from mlflow import log_metric, log_param, log_artifacts
from torch.nn.parallel import DistributedDataParallel as DDP
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

def finetune(gpu, args):
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
    
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    
    torch.cuda.set_device(local_rank)

    fsdp_params = dict(
    mixed_precision=False,
    flatten_parameters=True,
    state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
    cpu_offload=True,  # enable cpu offloading
)

    with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()

    # Wrap each layer in FSDP separately
    for name, child in model.named_children():
        if name == "layers":
            for layer_name, layer in child.named_children():
                wrapped_layer = wrap(layer)
                setattr(child, layer_name, wrapped_layer)
                
    model = wrap(model)
    
    
    
    dat = FastaBatchedDataset.from_file(args.query)
    # model_, alphabet = esm1_t12_85M_UR50S()





    dataset_set = set(dat)
    batch_labels_train, batch_strs_train, batch_tokens_train = batch_converter(list(dataset_set))
    train_inputs = {}
    train_inputs['input_ids'] = batch_tokens_train
    train_inputs['labels'] = train_inputs['input_ids'].clone.detach()


    class proteinDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        def __len__(self):
            return len(self.encodings['input_ids'])


    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        model = model.to(f"cuda:{local_rank}")
    else:
        return("Don't run this on a CPU")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dat,
        num_replicas=int(args.GPUs),
        rank=rank
    )

    dat_loader = torch.utils.data.DataLoader(dat, batch_size = int(args.batch_size), num_workers=0, sampler = train_sampler, collate_fn=alphabet.get_batch_converter())


    criterion = nn.CrossEntropyLoss(reduction='none').to(f"cuda:{local_rank}")

    ddp_model = model
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr = int(args.lr))

    ddp_model.train() 
    
    
#     tracker = LayerwiseMemoryTracker()
#     tracker.monitor(ddp_model)
    
    epochs = 20

    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(dat_loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optimizer.zero_grad()
            # GPUtil.showUtilization()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(f"cuda:{local_rank}")
            # attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(f"cuda:{local_rank}")
            # process
            output = model(input_ids, repr_layers = [-1])
            # extract loss
            loss = criterion(output['logits'].permute(0,2,1), labels)
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optimizer.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
        # store and save parameters and progress
        # torch.save(model, 'TrainingProgress_ep' + str(epoch) + ".pt")    
    
    
    
    
    
    
    

#     start = datetime.now()
#     optimizer.zero_grad()
#     for j in tqdm(range(int(args.epochs))):
#         for i, (labels, seq, toks) in enumerate(dat_loader):
#             del labels, seq
            
#             torch.cuda.empty_cache()

#             if toks.shape[1] > 1024:
#                 toks = torch.narrow(toks, 1, 0, 1023)

#             true_aa, masked_batch_tokens = prepare_mlm_mask(alphabet, toks)
#             optimizer.zero_grad()
#             if torch.cuda.is_available():
#                 masked_batch_tokens = masked_batch_tokens.to(f"cuda:{local_rank}", non_blocking=True)
#                 results = ddp_model(tokens = masked_batch_tokens, repr_layers=[12])
    
#             else:
#                 results = model_(masked_batch_tokens.to('cpu'), repr_layers=[12])
                
#             pred = results['logits']
#             loss = criterion(pred.permute(0,2,1).to(f"cuda:{local_rank}"),true_aa.to(f"cuda:{local_rank}"))
#             loss = loss.mean().sum()

# #             tracker.show_plots()

#             loss.backward()
#             #log_metric('CrossEntropyLoss', loss)
            

#             optimizer.step()                            
#             optimizer.zero_grad()
        
#         perplexity = torch.exp(loss)
#         print(f"Epoch: {j}")
#         print(f"Loss: {loss}")
#         # print(f"Perplexity: {perplexity}")
        
    states = ddp_model.state_dict()
    if rank == 0:

        torch.save(
            states,
            f"{args.name}_esm2t30.pt"
        )
    
    dist.destroy_process_group()
#     tracker.stop()
    # print("Training complete in: " + str(datetime.now() - start))

    return True
