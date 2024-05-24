# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import math

import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
import numpy as np
from scipy.spatial import transform
from scipy.stats import special_ortho_group
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from typing import Sequence, Tuple, List
import os
from icecream import ic
from esm.data import BatchConverter


def load_structure(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith('cif'):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith('pdb') or fpath.endswith('3di'):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    bbmask = filter_backbone(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure


def extract_coords_from_structure(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return coords, seq


def load_coords(fpath, chain):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    structure = load_structure(fpath, chain)
    return extract_coords_from_structure(structure)


def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "CA", "C"]
    """
    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)


def get_sequence_loss(model, alphabet, coords, seq):
    device = next(model.parameters()).device
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, seq)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)

    prev_output_tokens = tokens[:, :-1].to(device)
    target = tokens[:, 1:]
    target_padding_mask = (target == alphabet.padding_idx)
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
    loss = F.cross_entropy(logits, target, reduction='none')
    loss = loss[0].cpu().detach().numpy()
    target_padding_mask = target_padding_mask[0].cpu().numpy()
    return loss, target_padding_mask


def score_sequence(model, alphabet, coords, seq):
    loss, target_padding_mask = get_sequence_loss(model, alphabet, coords, seq)
    ll_fullseq = -np.sum(loss * ~target_padding_mask) / np.sum(~target_padding_mask)
    # Also calculate average when excluding masked portions
    coords = coords.numpy()
    coord_mask = np.all(np.isfinite(coords), axis=(-1, -2))
    ll_withcoord = -np.sum(loss * coord_mask) / np.sum(coord_mask)
    return ll_fullseq, ll_withcoord


def get_encoder_output(model, alphabet, coords):
    device = next(model.parameters()).device
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, seq)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)
    encoder_out = model.encoder.forward(coords, padding_mask, confidence,
            return_all_hiddens=False)
    # remove beginning and end (bos and eos tokens)
    return encoder_out['encoder_out'][0][1:-1, 0]


def rotate(v, R):
    """
    Rotates a vector by a rotation matrix.
    
    Args:
        v: 3D vector, tensor of shape (length x batch_size x channels x 3)
        R: rotation matrix, tensor of shape (length x batch_size x 3 x 3)

    Returns:
        Rotated version of v by rotation matrix R.
    """
    R = R.unsqueeze(-3)
    v = v.unsqueeze(-1)
    return torch.sum(v * R, dim=-2)


def get_rotation_frames(coords):
    """
    Returns a local rotation frame defined by N, CA, C positions.

    Args:
        coords: coordinates, tensor of shape (batch_size x length x 3 x 3)
        where the third dimension is in order of N, CA, C

    Returns:
        Local relative rotation frames in shape (batch_size x length x 3 x 3)
    """
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-2)
    return R


def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)


def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    v_expand = torch.unsqueeze(values, -1)
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)


def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(
            torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)


def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )


class CoordBatchConverter(BatchConverter):
    def __call__(self, raw_batch: Sequence[Tuple[Sequence, str]], device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x 3 x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>") 
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))

        coords_and_confidence, strs, tokens = super().__call__(batch)

        # pad beginning and end of each protein due to legacy reasons
        coords = [
            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.inf)
            for cd, _ in coords_and_confidence
        ]
        confidence = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.)
            for _, cf in coords_and_confidence
        ]
        coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
        padding_mask = torch.isnan(coords[:,:,0,0])
        coord_mask = torch.isfinite(coords.sum(-2).sum(-1))
        confidence = confidence * coord_mask + (-1.) * padding_mask
        return coords, confidence, strs, tokens, padding_mask

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result

def download_ligmpnn_weights(directory):
    base_url = "https://files.ipd.uw.edu/pub/ligandmpnn/"
    file_names = [
        "ligandmpnn_v_32_005_25.pt",
        "ligandmpnn_v_32_010_25.pt",
        "ligandmpnn_v_32_020_25.pt",
        "ligandmpnn_v_32_030_25.pt",
        "per_residue_label_membrane_mpnn_v_48_020.pt",
        "global_label_membrane_mpnn_v_48_020.pt",
        "solublempnn_v_48_002.pt",
        "solublempnn_v_48_010.pt",
        "solublempnn_v_48_020.pt",
        "solublempnn_v_48_030.pt",
        "ligandmpnn_sc_v_32_002_16.pt",
        "proteinmpnn_v_48_002.pt",
        "proteinmpnn_v_48_010.pt",
        "proteinmpnn_v_48_020.pt",
        "proteinmpnn_v_48_030.pt"
    ]
    
    # Create the directory if it does not exist
    weights_dir = os.path.join(directory, "LigandMPNN_weights")
    os.makedirs(weights_dir)
    
    # Download each file
    for file_name in file_names:
        file_path = os.path.join(weights_dir, file_name)
        if not os.path.exists(file_path):  # Check if the file already exists before downloading
            os.system(f"wget -q {base_url}{file_name} -O {file_path}")

    return

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:27:44 2023
https://github.com/mheinzinger/ProstT5/blob/main/scripts/predict_3Di_encoderOnly.py
@author: mheinzinger
"""

import argparse
import time
from pathlib import Path

from urllib import request
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer


if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device: {}".format(device))


# Convolutional neural network (two convolutional layers)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        """
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of classes (20 for 3Di)
        """
        x = x.permute(0, 2, 1).unsqueeze(
            dim=-1)  # IN: X = (B x L x F); OUT: (B x F x L, 1)
        Yhat = self.classifier(x)  # OUT: Yhat_consurf = (B x N x L x 1)
        Yhat = Yhat.squeeze(dim=-1)  # IN: (B x N x L x 1); OUT: ( B x N x L )
        return Yhat


def get_T5_model(model_dir):
    print("Loading T5 from: {}".format(model_dir))
    model = T5EncoderModel.from_pretrained(
        "Rostlab/ProstT5_fp16", cache_dir=model_dir).to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(
        "Rostlab/ProstT5_fp16", do_lower_case=False, cache_dir=model_dir)
    return model, vocab


def read_fasta(fasta_path, split_char, id_field):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''

    sequences = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace(
                    '>', '').strip().split(split_char)[id_field]
                sequences[uniprot_id] = ''
            else:
                s = ''.join(line.split()).replace("-", "")

                if s.islower():  # sanity check to avoid mix-up of 3Di and AA input
                    print("The input file was in lower-case which indicates 3Di-input." +
                          "This predictor only operates on amino-acid-input (upper-case)." +
                          "Exiting now ..."
                          )
                    return None
                else:
                    sequences[uniprot_id] += s
    return sequences


def write_probs(predictions, out_path):
    with open(out_path, 'w+') as out_f:
        out_f.write('\n'.join(
            ["{},{}".format(seq_id, prob)
             for seq_id, (N, prob) in predictions.items()
             ]
        ))
    print(f"Finished writing probabilities to {out_path}")
    return None


def write_predictions(predictions, out_path):
    ss_mapping = {
        0: "A",
        1: "C",
        2: "D",
        3: "E",
        4: "F",
        5: "G",
        6: "H",
        7: "I",
        8: "K",
        9: "L",
        10: "M",
        11: "N",
        12: "P",
        13: "Q",
        14: "R",
        15: "S",
        16: "T",
        17: "V",
        18: "W",
        19: "Y"
    }

    with open(out_path, 'w+') as out_f:
        out_f.write('\n'.join(
            [">{}\n{}".format(
                seq_id, "".join(list(map(lambda yhat: ss_mapping[int(yhat)], yhats))))
             for seq_id, (yhats, _) in predictions.items()
             ]
        ))
    print(f"Finished writing results to {out_path}")
    return None


def toCPU(tensor):
    if len(tensor.shape) > 1:
        return tensor.detach().cpu().squeeze(dim=-1).numpy()
    else:
        return tensor.detach().cpu().numpy()


def download_file(url, local_path):
    # if not local_path.parent.is_dir():
    #     local_path.parent.mkdir()

    print("Downloading: {}".format(url))
    req = request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
    })

    with request.urlopen(req) as response, open(local_path, 'wb') as outfile:
        shutil.copyfileobj(response, outfile)
    return None


def load_predictor(cache_dir, weights_link="https://github.com/mheinzinger/ProstT5/raw/main/cnn_chkpnt/model.pt"):
    model = CNN()
    checkpoint_p = os.path.join(cache_dir, "model.pt")
    # if no pre-trained model is available, yet --> download it
    if not os.path.exists(checkpoint_p):
        download_file(weights_link, checkpoint_p)

    # Torch load will map back to device from state, which often is GPU:0.
    # to overcome, need to explicitly map to active device
    global device

    state = torch.load(checkpoint_p, map_location=device)

    model.load_state_dict(state["state_dict"])

    model = model.eval()
    model = model.to(device)

    return model


def get_3di_prostt5(args, out_path, model_dir, split_char = '!', id_field = 0, half_precision = True, output_probs=True, max_residues=40000, max_seq_len=10000, max_batch=500):

    seq_dict = dict()
    predictions = dict()
    os.makedirs(os.path.join(out_path, f'{args.name}_ProstT5-3Di_outputs'), exist_ok=True)
    out_path = os.path.join(out_path, f'{args.name}_ProstT5-3Di_outputs')
    # Read in fasta
    seq_dict = read_fasta(args.query, split_char, id_field)
    prefix = "<AA2fold>"

    model, vocab = get_T5_model(model_dir)
    predictor = load_predictor(model_dir)

    if half_precision:
        model.half()
        predictor.half()
        print("Using models in half-precision.")
    else:
        model.to(torch.float32)
        predictor.to(torch.float32)
        print("Using models in full-precision.")

    # print('########################################')
    # print('Example sequence: {}\n{}'.format(next(iter(
    #     seq_dict.keys())), next(iter(seq_dict.values()))))
    # print('########################################')
    # print('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
    n_long = sum([1 for _, seq in seq_dict.items() if len(seq) > max_seq_len])
    # sort sequences by length to trigger OOM at the beginning
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(
        seq_dict[kv[0]]), reverse=True)

    # print("Average sequence length: {}".format(avg_length))
    # print("Number of sequences >{}: {}".format(max_seq_len, n_long))

    start = time.time()
    batch = list()
    standard_aa = "ACDEFGHIKLMNPQRSTVWY"
    standard_aa_dict = {aa: aa for aa in standard_aa}
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        # replace the non-standard amino acids with 'X'
        seq = ''.join([standard_aa_dict.get(aa, 'X') for aa in seq])
        #seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        seq_len = len(seq)
        seq = prefix + ' ' + ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs,
                                                     add_special_tokens=True,
                                                     padding="longest",
                                                     return_tensors='pt'
                                                     ).to(device)
            try:
                with torch.no_grad():
                    embedding_repr = model(token_encoding.input_ids,
                                           attention_mask=token_encoding.attention_mask
                                           )
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(
                    pdb_id, seq_len)
                )
                continue

            # ProtT5 appends a special tokens at the end of each sequence
            # Mask this also out during inference while taking into account the prefix
            for idx, s_len in enumerate(seq_lens):
                token_encoding.attention_mask[idx, s_len+1] = 0

            # extract last hidden states (=embeddings)
            residue_embedding = embedding_repr.last_hidden_state.detach()
            # mask out padded elements in the attention output (can be non-zero) for further processing/prediction
            residue_embedding = residue_embedding * \
                token_encoding.attention_mask.unsqueeze(dim=-1)
            # slice off embedding of special token prepended before to each sequence
            residue_embedding = residue_embedding[:, 1:]

            # IN: X = (B x L x F) - OUT: ( B x N x L )
            prediction = predictor(residue_embedding)
            if output_probs:  # compute max probabilities per token/residue if requested
                probabilities = toCPU(torch.max(
                    F.softmax(prediction, dim=1), dim=1, keepdim=True)[0])
            
            prediction = toCPU(torch.max(prediction, dim=1, keepdim=True)[
                               1]).astype(np.byte)

            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # slice off padding and special token appended to the end of the sequence
                pred = prediction[batch_idx, :, 0:s_len].squeeze()
                if output_probs:  # average over per-residue max.-probabilities
                    prob = int( 100* np.mean(probabilities[batch_idx, :, 0:s_len]))
                    predictions[identifier] = (pred, prob)
                else:
                    predictions[identifier] = (pred, None)
                assert s_len == len(predictions[identifier][0]), print(
                    f"Length mismatch for {identifier}: is:{len(predictions[identifier])} vs should:{s_len}")
                # if len(predictions) == 1:
                #     print(
                #         f"Example: predicted for protein {identifier} with length {s_len}: {predictions[identifier]}")

    end = time.time()
    print('\n############# STATS #############')
    print('Total number of predictions: {}'.format(len(predictions)))
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(
        end-start, (end-start)/len(predictions), avg_length))
    print("Writing results now to disk ...")
    write_predictions(predictions, f'{out_path}/{args.name}_ProstT5-3Di.fasta')
    if output_probs:
        write_probs(predictions, f'{out_path}/{args.name}_ProstT5-3Di_probs.csv')

    return True


# def create_arg_parser():
#     """"Creates and returns the ArgumentParser object."""

#     # Instantiate the parser
#     parser = argparse.ArgumentParser(description=(
#         'predict_3Di_encoderOnly.py translates amino acid sequences to 3Di sequences. ' +
#         'Example: python predict_3Di_encoderOnly.py --input /path/to/some_AA_sequences.fasta --output /path/to/some_3Di_sequences.fasta --model /path/to/tmp/checkpoint/dir')
#     )

#     # Required positional argument
#     parser.add_argument('-i', '--input', required=True, type=str,
#                         help='A path to a fasta-formatted text file containing protein sequence(s).')

#     # Required positional argument
#     parser.add_argument('-o', '--output', required=True, type=str,
#                         help='A path for saving the 3Di translations in FASTA format.')

#     # Required positional argument
#     parser.add_argument('--model', required=True, type=str,
#                         help='A path to a directory for saving the checkpoint of the pre-trained model.')

#     # Optional argument
#     parser.add_argument('--split_char', type=str,
#                         default='!',
#                         help='The character for splitting the FASTA header in order to retrieve ' +
#                         "the protein identifier. Should be used in conjunction with --id." +
#                         "Default: '!' ")

#     # Optional argument
#     parser.add_argument('--id', type=int,
#                         default=0,
#                         help='The index for the uniprot identifier field after splitting the ' +
#                         "FASTA header after each symbole in ['|', '#', ':', ' ']." +
#                         'Default: 0')

#     parser.add_argument('--half', type=int,
#                         default=1,
#                         help="Whether to use half_precision or not. Default: 1 (half-precision)")
    
#     parser.add_argument('--output_probs', type=int,
#                         default=1,
#                         help="Whether to output probabilities/reliability. Default: 1 (output them).")

#     return parser


# def main():
#     parser = create_arg_parser()
#     args = parser.parse_args()

#     seq_path = Path(args.input)  # path to input FASTAS
#     out_path = Path(args.output)  # path where predictions should be written to
#     model_dir = args.model  # path/repo_link to checkpoint

#     if out_path.is_file():
#         print("Output file is already existing and will be overwritten ...")

#     split_char = args.split_char
#     id_field = args.id

#     half_precision = False if int(args.half) == 0 else True
#     assert not (half_precision and device == "cpu"), print(
#         "Running fp16 on CPU is not supported, yet")
    
#     output_probs = False if int(args.output_probs) == 0 else True

#     get_embeddings(
#         seq_path,
#         out_path,
#         model_dir,
#         split_char,
#         id_field,
#         half_precision,
#         output_probs,
#     )


# if __name__ == '__main__':
#     main()
