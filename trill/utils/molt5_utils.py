# Utils for embedding small-molecules with MolT5
# Directly adapted from CataPro/inference/utils.py

from transformers import T5EncoderModel, T5Tokenizer
import torch
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from icecream import ic
from .poolparti import poolparti_gen

molt5_key = {
    'MolT5-Small':'molt5-small',
    'MolT5-Base':'molt5-base',
    'MolT5-Large':'molt5-large'
}

def prep_input_from_smiles_fasta(args):
    headers = []
    smiles_entries = []
    current_smiles = []
    current_header = None

    with open(args.query, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    smiles_entries.append("".join(current_smiles))
                    headers.append(current_header)
                    current_smiles = []
                current_header = line[1:]
            else:
                current_smiles.append(line)
        
        # Save the last entry
        if current_header is not None:
            smiles_entries.append("".join(current_smiles))
            headers.append(current_header)

    return headers, smiles_entries

def make_embedding_dataframe(headers, avg_embs):
    # Convert list of shape (1, 512) arrays to shape (n, 512)
    stacked = np.vstack([emb.squeeze() for emb in avg_embs])  # shape: (n, 512)

    # Create DataFrame
    df = pd.DataFrame(stacked, columns=[f"{i}" for i in range(stacked.shape[1])])

    # Add headers
    df["Label"] = headers

    return df
# def make_embedding_dataframe(headers, avg_embs):
#     ic([arr.shape for arr in avg_embs])

#     ic("=== make_embedding_dataframe ===")
#     ic(type(avg_embs))

#     try:
#         ic(len(avg_embs))
#     except Exception as e:
#         ic("len() failed:", e)

#     try:
#         ic(avg_embs.shape)
#     except Exception as e:
#         ic("avg_embs has no .shape:", e)

#     if isinstance(avg_embs, list):
#         ic("avg_embs is a list")
#         if len(avg_embs) > 0:
#             ic(type(avg_embs[0]))
#             try:
#                 ic(avg_embs[0].shape)
#             except Exception as e:
#                 ic("avg_embs[0] has no .shape:", e)

#             if hasattr(avg_embs[0], 'ndim'):
#                 ic("avg_embs[0].ndim:", avg_embs[0].ndim)
#         else:
#             ic("avg_embs is an empty list")

#     if hasattr(avg_embs, 'ndim'):
#         ic("avg_embs.ndim:", avg_embs.ndim)

#     try:
#         n_dims = avg_embs.shape[1]
#         ic(n_dims)
#     except Exception as e:
#         ic("avg_embs.shape[1] failed:", e)
#         raise

#     try:
#         df = pd.DataFrame(avg_embs, columns=list(range(n_dims)))
#         df["Label"] = headers
#         ic("DataFrame created successfully")
#         ic(df.shape)
#         ic(df.head())
#         return df
#     except Exception as e:
#         ic("DataFrame creation failed:", e)
#         raise
def get_molT5_embed(args):
    logger.info(f'Embedding small-molecule SMILES with {args.model}')
    tokenizer = T5Tokenizer.from_pretrained(f'laituan245/{molt5_key[args.model]}')
    model = T5EncoderModel.from_pretrained(f'laituan245/{molt5_key[args.model]}', use_safetensors=True)
    headers_list, smiles_list = prep_input_from_smiles_fasta(args)
    N_smiles = len(smiles_list)

    averaged_embeddings = []
    token_level_embeddings = []
    poolparti_embeddings = []
    all_attentions = []


    for smile in tqdm(smiles_list):
        input_ids = tokenizer(smile, return_tensors="pt").input_ids
        outputs = model(input_ids=input_ids, output_attentions=True, output_hidden_states=True)
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states[:, 1:-1]

        embed_avg = torch.mean(last_hidden_states[0], dim=0).detach().cpu().numpy()
        embed_tokens = last_hidden_states[0].detach().cpu().numpy()
        averaged_embeddings.append(embed_avg.reshape(1, -1))
        token_level_embeddings.append(embed_tokens)

        attn_mean_pooled_layers = []
        attn_max_pooled_layers = []
        num_hidden_layers = model.config.num_layers

        for layer in range(num_hidden_layers):
            attn_raw = outputs.attentions[layer] 
            attn_raw = attn_raw.squeeze(0)
            attn_mean_pooled_layers.append(torch.mean(attn_raw, dim=0))  
            attn_max_pooled_layers.append(torch.max(attn_raw, dim=0).values) 

        combined_attention = torch.stack([
            torch.stack(attn_mean_pooled_layers), 
            torch.stack(attn_max_pooled_layers)
        ]).unsqueeze(1) 

        all_attentions.append(combined_attention)
        poolparti_embs = poolparti_gen(embed_tokens, combined_attention.cpu().detach())
        poolparti_embs = poolparti_embs.numpy()

        poolparti_embeddings.append(poolparti_embs.reshape(1, -1))

    poolparti_df = make_embedding_dataframe(headers_list, poolparti_embeddings)
    avg_df = make_embedding_dataframe(headers_list, averaged_embeddings)

    return avg_df, token_level_embeddings, headers_list, poolparti_df
