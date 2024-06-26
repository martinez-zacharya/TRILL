import glob
import itertools
import os
import re
from argparse import Namespace
from typing import Sequence, Tuple, List

import esm
import numpy as np
import pandas as pd
import torch
from esm.constants import proteinseq_toks
from tqdm import tqdm
from loguru import logger
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from typing import Iterator, List, Tuple
import math
import random
from tqdm import trange

from .inverse_folding.gvp_transformer import lightning_GVPTransformerModel
from .inverse_folding.multichain_util import extract_coords_from_complex, score_sequence_in_complex
from .inverse_folding.util import load_structure, score_sequence


class coordDataset(torch.utils.data.Dataset):
    def __init__(self, input):
        self.input = input
    def __getitem__(self, idx):
        coords, seq = self.input[idx]
        return coords, seq
    def __len__(self):
        return len(self.input)

def ESM_IF1_Wrangle(infile):
    structures = load_structure(infile)
    data = extract_coords_from_complex(structures)
    data = coordDataset([data])
    return data

def ESM_IF1(data, genIters, temp, GPUs):
    complex_flag = False
    model_data, _ = esm.pretrained._download_model_and_regression_data('esm_if1_gvp4_t16_142M_UR50')
    model, alphabet = load_model_and_alphabet_core('esm_if1_gvp4_t16_142M_UR50', model_data)
    # model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    sampled_seqs = [()]
    native_seq_scores = [()]
    for batch in data:
        coords, native_seq = batch
        chains = list(coords.keys())
        if len(chains) > 1:
            complex_flag = True
        loop_chain = tqdm(chains)
        loop_chain.set_description('Chains')
        for coord in coords:
            coords[coord] = coords[coord].squeeze(0)
        for chain in loop_chain:
            loop_gen_iters = tqdm(range(int(genIters)))
            loop_gen_iters.set_description('Generative Iterations')
            if complex_flag == False:
                if isinstance(native_seq[chain], list):
                    seq = native_seq[chain][0]
                else:
                    seq = native_seq[chain]
                n_ll, _ = score_sequence(
                    model, alphabet, coords[chain], seq)
            else:
                coords_4scoring = {}
                for k, v in coords.items():
                    coords_4scoring[k] = v.numpy()
                n_ll, _ = score_sequence_in_complex(
                        model, alphabet, coords_4scoring, chain, native_seq[chain][0]) 
                
            native_seq_scores.append(tuple([native_seq[chain], f'{chain}_{n_ll}']))
            if GPUs != 0:
                model = model.cuda()
                pass
            for i in loop_gen_iters:
                sampled_seq = sample_sequence_in_complex(model, coords, chain, temperature=temp)
                if complex_flag == False:
                    ll, _ = score_sequence(
                    model, alphabet, coords[chain], sampled_seq)

                else:
                    try:
                        coords_4scoring = {}
                        for k, v in coords.items():
                            coords_4scoring[k] = v.numpy()
                        ll, _ = score_sequence_in_complex(
                        model, alphabet, coords_4scoring, chain, sampled_seq)

                    except ValueError:
                        logger.warning(f'{sampled_seq} could not be scored.')
                        ll = 'NA'
                sampled_seqs.append(tuple([sampled_seq, f'{chain}_{ll}']))
    sample_df = pd.DataFrame(sampled_seqs)
    sample_df = sample_df.iloc[1: , :]
    native_seq_scores = pd.DataFrame(native_seq_scores).iloc[1: , :]
    return sample_df, native_seq_scores

def clean_embeddings(model_reps):
    newdf = pd.DataFrame(model_reps, columns = ['Embeddings', 'Label'])
    finaldf = newdf['Embeddings'].apply(pd.Series)
    finaldf['Label'] = newdf['Label']
    return finaldf

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
            remark=f"pTM = {outputs['ptm'].item()}, mean pLDDT = {outputs['plddt'].mean()}",
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def sample_sequence_in_complex(model, coords, target_chain_id, temperature=1.,
        padding_length=10):
    """
    From fair-esm repo
    Samples sequence for one chain in a complex.
    Args:
        model: An instance of the GVPTransformer model
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        padding_length: padding length in between chains
    Returns:
        Sampled sequence for the target chain
    """
    target_chain_len = coords[target_chain_id].shape[0]
    all_coords = _concatenate_coords(coords, target_chain_id)
    device = next(model.parameters()).device

    # Supply padding tokens for other chains to avoid unused sampling for speed
    padding_pattern = ['<pad>'] * all_coords.shape[0]
    for i in range(target_chain_len):
        padding_pattern[i] = '<mask>'
    sampled = model.sample(all_coords, partial_seq=padding_pattern,
            temperature=temperature, device = device)
    sampled = sampled[:target_chain_len]
    return sampled

def _concatenate_coords(coords, target_chain_id, padding_length=10):
    """
    From fair-esm repo

    Args:
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        padding_length: Length of padding between concatenated chains
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates, a
              concatenation of the chains with padding in between
            - seq is the extracted sequence, with padding tokens inserted
              between the concatenated chains
    """
    pad_coords = np.full((padding_length, 3, 3), np.nan, dtype=np.float32)
    # For best performance, put the target chain first in concatenation.
    coords_list = [coords[target_chain_id]]
    for chain_id in coords:
        if chain_id == target_chain_id:
            continue
        coords_list.append(pad_coords)
        coords_list.append(coords[chain_id])
    coords_concatenated = np.concatenate(coords_list, axis=0)
    return coords_concatenated

def _download_model_and_regression_data(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    model_data = load_hub_workaround(url)
    if _has_regression_weights(model_name):
        regression_data = load_regression_hub(model_name)
    else:
        regression_data = None
    return model_data, regression_data

def load_model_and_alphabet_core(model_name, model_data, regression_data=None):
    if regression_data is not None:
        model_data["model"].update(regression_data["model"])

    if model_name.startswith("esm2"):
        # model, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)
        pass
    else:
        model, alphabet, model_state = _load_model_and_alphabet_core_v1(model_data)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    if regression_data is None:
        expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
        error_msgs = []
        missing = (expected_keys - found_keys) - expected_missing
        if missing:
            error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
        unexpected = found_keys - expected_keys
        if unexpected:
            error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

        if error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        if expected_missing - found_keys:
            pass
            # warnings.warn(
            #     "Regression weights not found, predicting contacts will not produce correct results."
            # )

    model.load_state_dict(model_state, strict=regression_data is not None)

    return model, alphabet

def _load_model_and_alphabet_core_v1(model_data):
    import esm  # since esm.inverse_folding is imported below, you actually have to re-import esm here

    alphabet = esm.Alphabet.from_architecture(model_data["args"].arch)

    if "invariant_gvp" in model_data["args"].arch:
        # import esm.inverse_folding

        # model_type = GVPTransformerModel
        model_type = lightning_GVPTransformerModel
        model_args = vars(model_data["args"])  # convert Namespace -> dict

        def update_name(s):
            # Map the module names in checkpoints trained with internal code to
            # the updated module names in open source code
            s = s.replace("W_v", "embed_graph.embed_node")
            s = s.replace("W_e", "embed_graph.embed_edge")
            s = s.replace("embed_scores.0", "embed_confidence")
            s = s.replace("embed_score.", "embed_graph.embed_confidence.")
            s = s.replace("seq_logits_projection.", "")
            s = s.replace("embed_ingraham_features", "embed_dihedrals")
            s = s.replace("embed_gvp_in_local_frame.0", "embed_gvp_output")
            s = s.replace("embed_features_in_local_frame.0", "embed_gvp_input_features")
            return s

        model_state = {
            update_name(sname): svalue
            for sname, svalue in model_data["model"].items()
            if "version" not in sname
        }

    else:
        raise ValueError("Unknown architecture selected")

    model = model_type(
        Namespace(**model_args),
        alphabet,
    )

    return model, alphabet, model_state


def parse_and_save_all_predictions(args):
    # Look for all 'predictions_*.pt' files in the specified directory
    prediction_files = glob.glob(f"{args.outdir}/predictions_*.pt")
    
    # Initialize lists to hold parsed data across all files
    all_parsed_data_avg = []
    all_batched_per_aa_embeddings_with_labels = []  # Modified to store labels

    for file_path in prediction_files:
        # Load the predictions file
        preds = torch.load(file_path)
        
        # Initialize lists to hold parsed data
        parsed_data_avg = []
        
        # Initialize a list to hold per amino acid embeddings and labels for each batch
        batched_per_aa_embeddings_with_labels = []  # Modified to store labels

        # Start parsing
        for outer_list in preds:
            batch_per_aa_embeddings = []  # To hold per_AA embeddings for the current batch
            for tuple_in_outer_list in outer_list:
                if len(tuple_in_outer_list) == 2:
                    per_aa_list, avg_list = tuple_in_outer_list
                else:
                    # Handle cases where there's only one type of representation
                    if isinstance(tuple_in_outer_list[0][0], np.ndarray):
                        if len(tuple_in_outer_list[0][0].shape) > 1:
                            per_aa_list = tuple_in_outer_list
                            avg_list = []
                        else:
                            avg_list = tuple_in_outer_list
                            per_aa_list = []
                
                # Parsing per amino acid representations
                for tuple_in_per_aa_list in per_aa_list:
                    embedding, label = tuple_in_per_aa_list
                    batch_per_aa_embeddings.append((embedding, label))  # Modified to store labels
                    
                # Parsing average representations
                for tuple_in_avg_list in avg_list:
                    embedding, label = tuple_in_avg_list
                    parsed_data_avg.append((embedding.flatten(), label))
            
            # Append the batch of per_AA embeddings to the main list
            if batch_per_aa_embeddings:
                batched_per_aa_embeddings_with_labels.append(batch_per_aa_embeddings)  # Modified to store labels

        # Accumulate parsed data across all files
        all_parsed_data_avg.extend(parsed_data_avg)
        all_batched_per_aa_embeddings_with_labels.extend(batched_per_aa_embeddings_with_labels)  # Modified to store labels

    # Save average embeddings as CSV
    if all_parsed_data_avg:
        df_parsed_avg = pd.DataFrame(all_parsed_data_avg, columns=['Embeddings', 'Label'])
        finaldf_parsed_avg = df_parsed_avg['Embeddings'].apply(pd.Series)
        finaldf_parsed_avg['Label'] = df_parsed_avg['Label']
        if args.command == 'embed':
            outname = os.path.join(args.outdir, f'{args.name}_{args.model}_AVG.csv')
        elif args.command == 'classify':
            outname = os.path.join(args.outdir, f'{args.name}_ProtT5_AVG.csv')
        finaldf_parsed_avg.to_csv(outname, index=False)
    
    # Save batched per_AA embeddings as .pt file
    if all_batched_per_aa_embeddings_with_labels:  # Modified to store labels
        if args.command == 'embed':
            outname = os.path.join(args.outdir, f'{args.name}_{args.model}_perAA.pt')
            torch.save(all_batched_per_aa_embeddings_with_labels, outname)  # Modified to store labels

    return 

class premasked_FastaBatchedDataset(object):
    def __init__(self, sequence_labels, sequence_strs, sequence_masked):
        self.sequence_labels = sequence_labels
        self.sequence_strs = sequence_strs
        self.sequence_masked = sequence_masked

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        return self.sequence_labels[idx], self.sequence_strs[idx], self.sequence_masked[idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches

class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self, truncation_seq_length: int = None, masked=False):
        if masked:
            return masked_BatchConverter(self, truncation_seq_length)
        # else:
        #     return BatchConverter(self, truncation_seq_length)

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        if name in ("ESM-1", "protein_bert_base"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        elif name in ("ESM-1b", "roberta_large"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
            use_msa = False
        elif name in ("MSA Transformer", "msa_transformer"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = False
            use_msa = True
        elif "invariant_gvp" in name.lower():
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>", "<cath>", "<af2>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        else:
            raise ValueError("Unknown architecture selected")
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


class masked_BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list, seq_masked_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_masked_list]
        seq_actual_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        if self.truncation_seq_length:
            seq_encoded_list = [seq_str[:self.truncation_seq_length] for seq_str in seq_encoded_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        actual_tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        masked_tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        actual_tokens.fill_(self.alphabet.padding_idx)
        masked_tokens.fill_(self.alphabet.padding_idx)

        labels = []
        strs = []

        for i, (label, actual_seq_encoded, seq_encoded) in enumerate(
            zip(batch_labels, seq_actual_encoded_list, seq_encoded_list)
        ):
            labels.append(label)
            # strs.append(seq_str)
            if self.alphabet.prepend_bos:
                actual_tokens[i, 0] = self.alphabet.cls_idx
                masked_tokens[i, 0] = self.alphabet.cls_idx

            masked_seq = torch.tensor(seq_encoded, dtype=torch.int64)
            actual_seq = torch.tensor(actual_seq_encoded, dtype=torch.int64)
            masked_tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = masked_seq
            actual_tokens[
                i,
                int(self.alphabet.prepend_bos) : len(actual_seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = actual_seq
            if self.alphabet.append_eos:
                actual_tokens[i, len(actual_seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
                masked_tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx


        return labels, actual_tokens, masked_tokens


# class MSABatchConverter(BatchConverter):
    # def __call__(self, inputs: Union[Sequence[RawMSA], RawMSA]):
    #     if isinstance(inputs[0][0], str):
    #         # Input is a single MSA
    #         raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
    #     else:
    #         raw_batch = inputs  # type: ignore

    #     batch_size = len(raw_batch)
    #     max_alignments = max(len(msa) for msa in raw_batch)
    #     max_seqlen = max(len(msa[0][1]) for msa in raw_batch)

    #     tokens = torch.empty(
    #         (
    #             batch_size,
    #             max_alignments,
    #             max_seqlen + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
    #         ),
    #         dtype=torch.int64,
    #     )
    #     tokens.fill_(self.alphabet.padding_idx)
    #     labels = []
    #     strs = []

    #     for i, msa in enumerate(raw_batch):
    #         msa_seqlens = set(len(seq) for _, seq in msa)
    #         if not len(msa_seqlens) == 1:
    #             raise RuntimeError(
    #                 "Received unaligned sequences for input to MSA, all sequence "
    #                 "lengths must be equal."
    #             )
    #         msa_labels, msa_strs, msa_tokens = super().__call__(msa)
    #         labels.append(msa_labels)
    #         strs.append(msa_strs)
    #         tokens[i, : msa_tokens.size(0), : msa_tokens.size(1)] = msa_tokens

    #     return labels, strs, tokens


def read_fasta(
    path,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    with open(path, "r") as f:
        for result in read_alignment_lines(
            f, keep_gaps=keep_gaps, keep_insertions=keep_insertions, to_upper=to_upper
        ):
            yield result


def read_alignment_lines(
    lines,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    seq = desc = None

    def parse(s):
        if not keep_gaps:
            s = re.sub("-", "", s)
        if not keep_insertions:
            s = re.sub("[a-z]", "", s)
        return s.upper() if to_upper else s

    for line in lines:
        # Line may be empty if seq % file_line_width == 0
        if len(line) > 0 and line[0] == ">":
            if seq is not None:
                yield desc, parse(seq)
            desc = line.strip().lstrip(">")
            seq = ""
        else:
            assert isinstance(seq, str)
            seq += line.strip()
    assert isinstance(seq, str) and isinstance(desc, str)
    yield desc, parse(seq)

def premasked_finetune():
    pass

## from https://github.com/seanrjohnson/protein_gibbs_sampler/blob/main/src/pgen/esm_sampler.py

def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, valid_idx=None):
    """ Generate a word from from out[gen_idx]
    
    args:
        - out (torch.Tensor): tensor of logits of size seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Otherwise sample from top_k amino acids
        - valid_idx (list): list of valid indexes to return. If none, all indexes are valid
    returns:
        tensor containing the selected amino acid index
    """
    #TODO: repetition penalty.
    #TODO: this could be vectorized a lot better, but I think this isn't the rate limiting step (inferrence is), so it probably doesn't matter.

    logits = out[gen_idx] # 1 x vocab_size
    if temperature is not None:
        logits = logits / temperature

    if valid_idx is None:
        valid_idx = list(range(len(logits)))

    sub_logits = logits[valid_idx]

    if sample or (top_k <= 0) or (top_k > len(sub_logits)):
        # If sample is true, that means we are forcing sampling from the whole distribution.
        # If top_k is 0 that means we want to sample from the whole distribution.
        top_k = len(sub_logits)
    else:
        # top_k is in bounds and we aren't forcing full sampling, so just keep it as it is.
        top_k = top_k

    kth_vals, kth_idx = sub_logits.topk(top_k)  # kth_vals is the logits, kth_idx is the indexes at which the logits are found.
    dist = torch.distributions.categorical.Categorical(logits=kth_vals)

    idx = kth_idx[dist.sample()]

    return torch.tensor(valid_idx[idx])

class ESM1v():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm1v_t33_650M_UR90S()
        self.batch_converter = self.alphabet.get_batch_converter()

class ESM2_650M():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()

class ESM2_150M():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()

ESM_ALLOWED_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

class ESM_sampler():
    """adapted from bert-gen bert-babble.ipynb"""


    def __init__(self, model, device="cpu"):
        """
            model should be an object with parameters model, alphabet, and batch_converter
        """
        self.model = model

        #switch model to eval mode
        #TODO: CHECK THAT THIS ACTUALLY SWITCHES THE MODEL TO EVAL MODE AND TURNS OFF GRADIENTS!
        self.model.model = self.model.model.eval()
        self.cuda = False
        #set device
        #TODO: handle case where there are multiple cuda devices.
        if (device == "gpu"):
            if (torch.cuda.is_available()):
                device = 'cuda'
                self.cuda = True
            else:
                raise(Exception("gpu requested, but No Cuda devices found"))
        else:
            device = "cpu"
        device = torch.device(device)
        self.model.model.to(device)

        self.valid_aa_idx = sorted([self.model.alphabet.get_idx(tok) for tok in ESM_ALLOWED_AMINO_ACIDS])

    def untokenize_batch(self, batch, bos, eos): #TODO: maybe should be moved to the model class, or a model superclass?
        #convert tokens to AAs, but skip the first one, because that one is <cls>
        start_offset = 0
        end_offset = 0
        if bos:
            start_offset = 1
        if eos:
            end_offset = -1
        out = [ "".join([self.model.alphabet.get_tok(seq[i]) for i in range(0 + start_offset, len(seq) + end_offset) ]) for seq in batch]
        return out

    @staticmethod
    def clean_seed_seq(seed_to_clean):
        cleaned_seq = seed_to_clean.upper()
        input_chars = {s for s in cleaned_seq}
        valid_chars = {s for s in ESM_ALLOWED_AMINO_ACIDS}
        if not input_chars.issubset(valid_chars):
            raise (Exception("Invalid input character: " + ",".join(input_chars-valid_chars)))
        return cleaned_seq

    def get_init_seq(self, seed_seq, max_len, batch_size = 1):
        """ Get initial sequence by padding seed_seq with masks """
        # In the BertGen paper they talk about padding with random sequence. I'm not sure that's a good idea. S.R.J.
        # Also, that code was commented out in the BertGen repo. So they probably didn't think was a good idea either.



        if isinstance(seed_seq, list): # input is an array, convert it to a string
            batch = random.choices(seed_seq, k=batch_size)
            for i, seed in enumerate(batch):
                remaining_len = max_len - len(seed)
                batch[i] = (str(i), self.clean_seed_seq(seed) + "<mask>" * remaining_len)

        elif isinstance(seed_seq, str):
            remaining_len = max_len - len(seed_seq)
            seed_seq = self.clean_seed_seq(seed_seq)
            batch = [(str(i), seed_seq + "<mask>" * remaining_len) for i in range(batch_size)]

        else:
            raise (Exception("seed sequence should either be a string or list"))

        labels, strs, tokens = self.model.batch_converter(batch)
        return tokens

    def generate(self, n_samples, seed_seq, batch_size=1, in_order=False, max_len=None, leader_length=0, leader_length_percent=None, top_k=0, temperature=None, num_iters=10,  burnin=float('inf'),
                            mask=True, num_positions=0, num_positions_percent=None, indexes=None, rollover_from_start=False, show_progress_bar=True):
        """ generate sequences

            n_samples: number of sequences to output
            seed_seq: protein msa to start from
            batch_size: how many copies of the seed msa to run at one time.
            in_order: if True then cycle through the positions in order, otherwise randomly select positions each iteration.
            max_len: maximum size of each generated sequence. If None, then use the length of the longest input msa.
            leader_length: don't overwrite this many amino acids at the beginning of the sequence.
            leader_length_percent: if not None, then will set leader_length = int(len(seed_seq)*(leader_length_percent / 100))
            top_k: if >0, only sample from the top k most probable AAs
            temperature: higher numbers will mean there is a lower penalty for low-scoring amino acids.
            num_iters: how many times to run the forward loop for every batch. 
            burnin: during burn-in period, sample from full distribution; afterwards sample from top_k, set to 0 to never sample from full distribution (always take from top_k), or inf to always sample from full distribution.

            num_positions: generate new AAs for this many positions each iteration. If 0, then generate for all target positions each round.
            num_positions_percent: If not None, then set num_positions = int(len(seed_seq)*(num_positions_percent / 100))
            indexes: positions of the input sequence to modify. 1-indexed, if None then all positions after the leader.

            show_progress_bar: if True then show a progress bar corresponding to the number of batches that need to be processed. Default: True.

            #### Examples #####
            seed = "MTSENPLLALREKISALDEKLLALLAERRELAVEVGKAKLLSHRPVRDIDRERDLLERLITLGKAHHLDAHYITRLFQLIIEDSVLTQQALLQQH"

            #To generate AAs one position at a time in order:
                sampler.generate(n_samples=1, seed_seq=seed, batch_size=1, in_order=True, num_positions=1, num_iters=len(seed), mask=True)
            #To generate the entire protein at once:
                sampler.generate(n_samples=1, seed_seq=seed, batch_size=1, max_len=len(seed), in_order=True, num_positions=len(seed), num_iters=1, mask=False)
            #To go 15 iterations over the protein where a 10% of AAs randomly distributed through the protein are mutated on each iteration:
                sampler.generate(n_samples=1, seed_seq=seed, batch_size=1, max_len=len(seed), in_order=False, num_positions=int(len(seed)/10), num_iters=15, mask=False)
            #To go 15 iterations over the protein where a 10% of AAs randomly distributed through the protein are mutated on each iteration, and k=0 for the first 5 iterations, but k=1 for the remaining:
                sampler.generate(n_samples=1, seed_seq=seed, batch_size=1, max_len=len(seed), in_order=False, num_positions=int(len(seed)/10), num_iters=15, burnin=5, k=1, mask=False)
            
            #### Sequence Completion ####
            seed = "MTSENPLLALREKISALDEKLLALLAERRELAVE"
            product_length = 95

            #generate L->R one at a time
                out = sampler.generate(1, seed_seq=seed, batch_size=1, max_len=product_length, in_order=True, top_k=0, leader_length=len(seed), num_positions=1, num_iters=product_length-len(seed), mask=True)
            #generate all at a time
                out = sampler.generate(1, seed_seq=seed, batch_size=1, max_len=product_length, in_order=True, top_k=0, leader_length=len(seed), num_positions=product_length-len(seed), num_iters=1, mask=True)
        """

        #TODO: repetition penalty, somehow?
        #TODO: add dilated sequential sampling, like sampling every third or fifth amino acid and then doing the whole protein in like 3 or 5 steps, or something like that.
        with torch.no_grad(): # I'm not sure if this no_grad is necessary or not, but it couldn't hurt!
            if isinstance(seed_seq, str):
                sequence_length = len(seed_seq)
            elif isinstance(seed_seq, list):
                sequence_length = max(len(seed) for seed in seed_seq)
            else:
                raise ValueError("Unknown seed sequence format, expecting str or list")

            cuda = self.cuda
            sequences = []
            n_batches = math.ceil(n_samples / batch_size)

            if max_len is None:
                max_len = sequence_length

            if num_positions_percent is not None:
                num_positions = int(max_len*(num_positions_percent / 100))
            if num_positions < 0:
                num_positions = 0

            if leader_length_percent is not None:
                leader_length = int(max_len*(leader_length_percent / 100))
            if leader_length < 0:
                leader_length = 0

            for batch_n in trange(n_batches, disable=(not show_progress_bar)):

                batch = self.get_init_seq(seed_seq, max_len, batch_size)
                batch = batch.cuda() if cuda else batch

                indexes, last_i = self.calculate_indexes(indexes, leader_length, max_len, rollover_from_start)

                if num_positions > len(indexes):
                    num_positions = len(indexes)

                for ii in range(num_iters):
                    if num_positions > 0: #do some subset of positions
                        if in_order: #cycle through the indexes
                            next_i = last_i
                            last_i, target_indexes = self.get_target_index_in_order(batch_size, indexes, next_i,
                                                                                    num_positions)
                        else:
                            target_indexes = self.get_random_target_index(batch_size, indexes, num_positions)
                    else:
                        target_indexes = [indexes] * batch_size

                    if mask:
                        self.mask_target_indexes(batch, target_indexes)

                    out = self.model.model(batch)["logits"]

                    for batch_index in range(batch_size):
                        for kk in target_indexes[batch_index]:
                            idx = generate_step(out[batch_index],
                                                gen_idx=kk,
                                                top_k=top_k,
                                                temperature=temperature,
                                                sample=(ii < burnin),
                                                valid_idx=self.valid_aa_idx)

                            batch[batch_index][kk] = idx

                if batch_n == (n_batches - 1): #last batch, so maybe don't take all of them, just take enough to get to n_samples
                    sequences += self.untokenize_batch(batch, self.model.alphabet.prepend_bos, self.model.alphabet.append_eos)[0:n_samples - len(sequences)]
                else:
                    sequences += self.untokenize_batch(batch, self.model.alphabet.prepend_bos, self.model.alphabet.append_eos)
            return sequences

    def get_random_target_index(self, batch_size, indexes, num_positions):
        target_indexes = list()
        for b in range(batch_size):
            target_indexes.append(random.sample(indexes, num_positions))
        return target_indexes

    def get_target_index_in_order(self, batch_size, indexes, next_i, num_positions):
        sampled = 0
        target_indexes = list()
        while sampled < num_positions:
            sampled += 1
            next_i = (next_i + 1) % len(indexes)
            target_indexes.append(indexes[next_i])
        target_indexes = [target_indexes] * batch_size
        last_i = next_i
        return last_i, target_indexes

    def mask_target_indexes(self, batch, target_indexes):
        for batch_index in range(len(target_indexes)):
            for kk in target_indexes[batch_index]:
                batch[batch_index][kk] = self.model.alphabet.mask_idx

    def calculate_indexes(self, indexes, leader_length, max_len, rollover_from_start):
        if indexes is None:
            indexes = range(1, max_len + 1)  # skip position 1, because that should be <cls>
            if not rollover_from_start:  # we rollover from the end of the leader sequence
                indexes = indexes[leader_length:]
                last_i = leader_length - 1
            else:
                last_i = -1
        else:
            last_i = -1
        return indexes, last_i


    def log_likelihood(self, seq, with_masking=True, verbose=False, mask_distance=float("inf"), batch_size=None, device = 'cpu'):
        """
            seq: a protein sequence string
            with_masking: if True, then iterate over the sequence masking one position at a time and summing the log likelihoods of the correct choice at the masked positions.
                        if False, then run the model just once, on the unmasked sequence.
            mask_distance: For optimization, when masking individual positions, the distance between masked positions in the same execution, by default only one position is masked per model call.
            batch_size: number of MSAs to run on the gpu at once, if None, then batch_size=len(msa_list). default=None.
        """
        return self.log_likelihood_batch([seq], with_masking, verbose, mask_distance, batch_size)[0]
        
    #TODO: convert to iterator
    def log_likelihood_batch(self, seq_list, with_masking=True, verbose=False, mask_distance=float("inf"), batch_size=None, device='cpu'):

        # TODO: Allow batching to calculate likelihoods for multiple sequences at a time (how does padding effect likelihoods for sequences shorter than the longest sequence, hopefully not at all).

        # Inspired by and borrowing code from:
        # https://github.com/facebookresearch/esm/blob/master/variant-prediction/predict.py

        results = []

        for seq in tqdm(seq_list):
            reformatted_seq = [(str(0), self.clean_seed_seq(seq))]
            _, _, tokens = self.model.batch_converter(reformatted_seq)

            range_start = 1 if self.model.alphabet.prepend_bos else 0
            end_modifier = -1 if self.model.alphabet.append_eos else 0

            batch_range_end = len(seq) + range_start
            overall_range_end = tokens.shape[1] + end_modifier

            assert len(seq) == len(range(range_start, overall_range_end))

            tokens = tokens.cuda() if self.cuda else tokens
            with torch.no_grad():
                with torch.autocast(device_type=device):
                    token_probs = torch.log_softmax(self.model.model(tokens)['logits'], dim=-1)
                    log_likelihood_sum = 0.0
                    log_likelihood_list = []
                    for idx in range(range_start, batch_range_end):
                        likelihood = token_probs[0, idx, tokens[0, idx].item()]
                        log_likelihood_sum += likelihood
                        log_likelihood_list.append(likelihood.item())
                    results.append(float(log_likelihood_sum / len(seq)))

        return results