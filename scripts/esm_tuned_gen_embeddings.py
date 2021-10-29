from Bio import SeqIO
import pandas as pd
import numpy as np
import glob
import re
import requests
import io
import torch
from argparse import Namespace
from esm.constants import proteinseq_toks
import math
import torch.nn as nn
import torch.nn.functional as F
from esm.modules import TransformerLayer, PositionalEmbedding  # noqa
from esm.model import ProteinBertModel
import esm

alphabet = esm.Alphabet.from_dict(proteinseq_toks)
model_name = "esm1_t12_85M_UR50S"
url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
if torch.cuda.is_available():
    print("cuda")
    model_data = torch.hub.load_state_dict_from_url(url, progress=False)
else:
    model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location=torch.device('cpu'))

pra = lambda s: ''.join(s.split('decoder_')[1:] if 'decoder' in s else s)
prs = lambda s: ''.join(s.split('decoder.')[1:] if 'decoder' in s else s)
model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
model_state_12 = torch.load("../data/esm_t12_85M_UR50S_vp1s_20211026.pt")
model_t12 = esm.ProteinBertModel(Namespace(**model_args), len(alphabet), padding_idx=alphabet.padding_idx)
model_t12.load_state_dict(model_state_12)

vp1s = pd.read_csv('../data/VP1s.csv')

print_every = 100
def generate_embedding_transformer_t12(model,batch_converter,dat,dat_name,out_dir,seq_col):
    # initialize network 
    model.cuda()
    sequence_embeddings = []
    for epoch in range(dat.shape[0]):
        data = [(dat.iloc[epoch, 1], dat.iloc[epoch, seq_col])]
        _, _, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = model(batch_tokens.to('cuda'), repr_layers=[12])
            # last layer
            token_embeddings = results["representations"][12]
            seq = dat.iloc[epoch,seq_col]
            sequence_embeddings.append(token_embeddings[0, 1:len(seq) + 1].mean(0).cpu().detach().numpy())
        if epoch % print_every == 0:
            print(f"At Epoch: %.2f"% epoch)
            print(seq)
    sequence_embeddings = np.array(sequence_embeddings)
    print(sequence_embeddings.shape)
    print(out_dir + '/' + dat_name + ".npy")
    np.save(out_dir + '/' + dat_name + ".npy", sequence_embeddings)
    return

batch_converter = alphabet.get_batch_converter()
out_dir = "../data/kif"
generate_embedding_transformer_t12(model_t12,batch_converter,vp1s,"vp1s",out_dir,seq_col = 1)

