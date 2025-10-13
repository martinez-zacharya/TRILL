import os
import re
import time
import json
import numpy as np
import sys
import esm
import itertools
import torch
from torch.utils.data import Dataset
from transformers import EsmTokenizer

sys.path.append(".")

# Taken straight from https://github.com/westlake-repl/SaProt/blob/main/utils/foldseek_util.py with minimal edits


# Get structural seqs from pdb file
def get_struc_seq(path,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_mask: bool = False,
                  plddt_threshold: float = 70.,
                  foldseek_verbose: bool = False) -> dict:
    """

    Args:
        foldseek: Binary executable file of foldseek

        path: Path to pdb file

        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.

        process_id: Process ID for temporary files. This is used for parallel processing.

        plddt_mask: If True, mask regions with plddt < plddt_threshold. plddt scores are from the pdb file.

        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

        foldseek_verbose: If True, foldseek will print verbose messages.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    # assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"PDB file not found: {path}"
    
    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"
    if foldseek_verbose:
        cmd = f"foldseek structureto3didescriptor --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    else:
        cmd = f"foldseek structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)
    
    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            # Mask low plddt
            if plddt_mask:
                plddts = extract_plddt(path)
                assert len(plddts) == len(struc_seq), f"Length mismatch: {len(plddts)} != {len(struc_seq)}"
                
                # Mask regions with plddt < threshold
                indices = np.where(plddts < plddt_threshold)[0]
                np_seq = np.array(list(struc_seq))
                np_seq[indices] = "#"
                struc_seq = "".join(np_seq)
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]
            
            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
    
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict


def extract_plddt(pdb_path: str) -> np.ndarray:
    """
    Extract plddt scores from pdb file.
    Args:
        pdb_path: Path to pdb file.

    Returns:
        plddts: plddt scores.
    """
    with open(pdb_path, "r") as r:
        plddt_dict = {}
        for line in r:
            line = re.sub(' +', ' ', line).strip()
            splits = line.split(" ")
            
            if splits[0] == "ATOM":
                # If position < 1000
                if len(splits[4]) == 1:
                    pos = int(splits[5])
                
                # If position >= 1000, the blank will be removed, e.g. "A 999" -> "A1000"
                # So the length of splits[4] is not 1
                else:
                    pos = int(splits[4][1:])
                
                plddt = float(splits[-2])
                
                if pos not in plddt_dict:
                    plddt_dict[pos] = [plddt]
                else:
                    plddt_dict[pos].append(plddt)
    
    plddts = np.array([np.mean(v) for v in plddt_dict.values()])
    return plddts


def preprocess_saprot(args):
    if args.query.endswith('.txt'):
        with open(args.query, 'r') as pdb_paths:
            lines = pdb_paths.readlines()
            pdb_paths = [line.strip() for line in lines]
            sa_seqs = [get_struc_seq(path) for path in pdb_paths]
    else:
        pdb_paths = [os.path.join(args.query, file) for file in os.listdir(args.query) if file.endswith('.pdb')]
        sa_seqs = [get_struc_seq(path) for path in pdb_paths]
    
    base_names = [os.path.basename(path) for path in pdb_paths]
    file_names = [os.path.splitext(base_name)[0] for base_name in base_names]

    return sa_seqs, file_names

class SaProt_Dataset(Dataset):
    def __init__(self, sa_seqs, labels):
        self.sequences = []
        self.labels = []
        for sa_seq, file_name in zip(sa_seqs, labels):
            for chain in sa_seq.keys():
                _, _, sa = sa_seq[chain]
                chain_name = f'{file_name}_{chain}' 
                self.sequences.append(sa)
                self.labels.append(chain_name)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        return label, sequence
    
class SaProt_Collator:
    def __init__(self):
        self.tokenizer = EsmTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2", use_safetensors=True)

    def __call__(self, batch):
        # Extract sequences from the batch
        sequences = [item[1]for item in batch]
        labels = [item[0] for item in batch]
        # Tokenize the sequences
        sa_tokenized = self.tokenizer(sequences, return_tensors="pt", padding=True)

        return sa_tokenized, labels