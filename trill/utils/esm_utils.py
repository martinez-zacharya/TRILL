import esm
from esm.inverse_folding.util import load_structure, extract_coords_from_structure
from esm.inverse_folding.multichain_util import extract_coords_from_complex, sample_sequence_in_complex
import torch
import pandas as pd
from tqdm import tqdm

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

def ESM_IF1(data, genIters, temp):
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    sampled_seqs = [()]
    for batch in data:
        coords, native_seq = batch
        chains = list(coords.keys())
        loop_chain = tqdm(chains)
        loop_chain.set_description('Chains')
        for coord in coords:
            coords[coord] = coords[coord].squeeze(0)
        for chain in loop_chain:
            loop_gen_iters = tqdm(range(int(genIters)))
            loop_gen_iters.set_description('Generative Iterations')
            for i in loop_gen_iters:
                sampled_seq = sample_sequence_in_complex(model, coords, chain, temperature=temp)
                sampled_seqs.append(tuple([sampled_seq, chain]))
    sample_df = pd.DataFrame(sampled_seqs)
    sample_df = sample_df.iloc[1: , :]
    return sample_df

def clean_embeddings(model_reps):
    newdf = pd.DataFrame(model_reps, columns = ['Embeddings', 'Label'])
    finaldf = newdf['Embeddings'].apply(pd.Series)
    finaldf['Label'] = newdf['Label']
    return finaldf
