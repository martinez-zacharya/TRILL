import esm
from trill.utils.inverse_folding.util import load_structure, extract_coords_from_structure, score_sequence
from trill.utils.inverse_folding.multichain_util import extract_coords_from_complex, score_sequence_in_complex
import torch
import numpy as np
import pandas as pd
from argparse import Namespace
from tqdm import tqdm
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from trill.utils.inverse_folding.gvp_transformer import GVPTransformerModel, lightning_GVPTransformerModel
import warnings

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
                        print(f'{sampled_seq} could not be scored.')
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

