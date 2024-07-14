from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
from tokenizers import Tokenizer
import torch
import pandas as pd
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import os
import logging
import re
from Bio import SeqIO
import numpy as np

# Many of these scripts are straight from https://github.com/hugohrban/ProGen2-finetuning/tree/main with little or no modifications

def _prepare_data(input_file_name: str, bidirectional: bool = False, ctrl_tag = '') -> list[str]:
    """
    Prepare data from the input fasta file.
    """
    seqs = SeqIO.parse(open(input_file_name, "r"), "fasta")
    parsed_seqs = []
    for s in seqs:
        parsed_seqs.append(f"<|{ctrl_tag}|>1{str(s.seq)}2")
        if bidirectional:
            parsed_seqs.append(f"<|{ctrl_tag}|>2{str(s.seq)[::-1]}1")
    return parsed_seqs


def prepare_data(args, bidirectional=False, ctrl_tag=''):

    if not 0 <= args.eval <= 1:
        raise ValueError("Train-test split ratio must be between 0 and 1.")

    train_data = []
    test_data = []
    if args.query.endswith('.csv'):
        query_df = pd.read_csv(args.query, names=['Path', 'Tag'])
        for ix, row in query_df.iterrows():
            data = _prepare_data(row['Path'], bidirectional, ctrl_tag=row['Tag'])
            np.random.shuffle(data)
            split_idx = int(len(data) * args.eval)
            train_data.extend(data[split_idx:])
            test_data.extend(data[:split_idx])
    else:
        data = _prepare_data(args.query, bidirectional, ctrl_tag=ctrl_tag)
        np.random.shuffle(data)
        split_idx = int(len(data) * args.eval)
        train_data.extend(data[split_idx:])
        test_data.extend(data[:split_idx])

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    base_name = os.path.basename(args.query)
    file_name, _ = os.path.splitext(base_name)

    with open(os.path.join(args.outdir, f"train_{args.model}_{file_name}.txt"), "w") as f:
        for line in train_data:
            f.write(line + "\n")

    with open(os.path.join(args.outdir, f"test_{args.model}_{file_name}.txt"), "w") as f:
        for line in test_data:
            f.write(line + "\n")
    
    return os.path.join(args.outdir, f"train_{args.model}_{file_name}.txt"), os.path.join(args.outdir, f"test_{args.model}_{file_name}.txt")

class Protein_dataset(Dataset):
    def __init__(self, lines: list[str], tokenizer: Tokenizer):
        self.lines = lines
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForLanguageModeling(tokenizer = self.tokenizer, mlm=False)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = self.tokenizer.encode(line)
        ids = torch.tensor(line.ids)
        collated = self.data_collator([ids])
        return(collated)


def init_new_embeddings(model, prefixes: list[str]):
    if len(prefixes) <= 2:
        return
    new_embs = torch.zeros((len(prefixes) - 2, model.config.embed_dim)).to(model.device)

    unk_token_emb: torch.Tensor = model.transformer.wte.weight[-1].detach()
    mean_unk_emb = torch.zeros_like(new_embs) + unk_token_emb.mean()
    std_unk_emb = torch.zeros_like(new_embs) + unk_token_emb.std()

    # initialize new embeddings with normal distribution same as untrained embeddings
    torch.normal(mean_unk_emb, std_unk_emb, out=new_embs)
    new_embs = torch.cat([model.transformer.wte.weight, new_embs], dim=0)
    model.transformer.wte.weight = torch.nn.Parameter(new_embs, requires_grad=True)
    model.config.vocab_size_emb = new_embs.shape[0]


def load_data(file: str) -> tuple[list[str], list[str]]:
    lines = []
    prefixes = set()
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            prefix = re.match(r"<\|.*\|>", line).group(0)
            prefixes.add(prefix)
            lines.append(line)
    prefixes = sorted(list(prefixes))
    return lines, prefixes

def create_deepspeed_config(args):

    if args.strategy == "deepspeed_stage_1":
        config = {
            "train_batch_size": int(args.batch_size),
            "train_micro_batch_size_per_gpu": int(args.batch_size),
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": float(args.lr),
                    "betas": "auto",
                    "eps": "auto",
                    "weight_decay": "auto"}
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1,
            }
        }
    elif args.strategy == "deepspeed_stage_2":
        config = {
            "train_batch_size": int(args.batch_size),
            "train_micro_batch_size_per_gpu": int(args.batch_size),
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": float(args.lr),
                    "betas": "auto",
                    "eps": "auto",
                    "weight_decay": "auto"}
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True,
                "round_robin_gradients": True
            }
        }

    elif args.strategy == "deepspeed_stage_2_offload":
        config = {
            "train_batch_size": int(args.batch_size),
            "train_micro_batch_size_per_gpu": int(args.batch_size),
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": float(args.lr),
                    "betas": "auto",
                    "eps": "auto",
                    "weight_decay": "auto"}
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True,
                "round_robin_gradients": True
            }
        }

    elif args.strategy == "deepspeed_stage_3":
        config = {
            "train_batch_size": int(args.batch_size),
            "train_micro_batch_size_per_gpu": int(args.batch_size),
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e5,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True,
            }
        }

    elif args.strategy == "deepspeed_stage_3_offload":
        config = {
            "zero_allow_untested_optimizer": True,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": float(args.lr),
                    "betas": "auto",
                    "eps": "auto",
                    "weight_decay": "auto"}
                
            },
            "bf16": {
                "enabled": True
            },
            "train_batch_size": int(args.batch_size),
            "train_micro_batch_size_per_gpu": int(args.batch_size),
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": False
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": False
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": 5e7,
                "stage3_prefetch_bucket_size": 5e5,
                "stage3_param_persistence_threshold": 1e3,
                "stage3_max_live_parameters": 1e3,
                "stage3_max_reuse_distance": 1e5,
                "stage3_gather_16bit_weights_on_model_save": True,
            }
        }
    
    elif not args.strategy:
        config = None

    return config