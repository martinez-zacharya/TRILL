# import esm
import torch
from argparse import Namespace
from esm.constants import proteinseq_toks
import math
import torch.nn as nn
import torch.nn.functional as F
from esm.modules import TransformerLayer, PositionalEmbedding  # noqa
from esm.model import ProteinBertModel

# model, alphabet = torch.hub.load("facebookresearch/esm", "esm1_t34_670M_UR50S")
import esm
from ych_util import prepare_mlm_mask
import pandas as pd
import time

dat = pd.read_csv("../data/VP1s.csv")
dat = dat.sample(frac = 0.25)
dat.head()

alphabet = esm.Alphabet.from_dict(proteinseq_toks)
# model_name = "esm1_t34_670M_UR50S"
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
model_state = {prs(arg[0]): arg[1] for arg in model_data["model"].items()}

model = esm.ProteinBertModel(
        Namespace(**model_args), len(alphabet), padding_idx=alphabet.padding_idx
    )

model.load_state_dict(model_state)
model.cuda()
model.train()

batch_converter = alphabet.get_batch_converter()
criterion = nn.CrossEntropyLoss()
lr = 0.0001 # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

start_time = time.time()
print_every = 300
for j in range(300):
    dat = dat.sample(frac = 1)
    for i in range(dat.shape[0]):
        if len(dat.iloc[i,1])>1024:
            seq = dat.iloc[i,1][:1023]
        else:
            seq = dat.iloc[i,1]
        lab = dat.iloc[i,0]
        data = [(lab, seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        true_aa,target_ind,masked_batch_tokens = prepare_mlm_mask(alphabet,batch_tokens)
        optimizer.zero_grad()
        results = model(masked_batch_tokens.to('cuda'), repr_layers=[34])   

        pred = results["logits"].squeeze(0)[target_ind,:]   
        target = true_aa.squeeze(0)
        loss = criterion(pred.cpu(),target)
        loss.backward()
        optimizer.step()

        if i % print_every == 1:
            print(batch_labels)
            print(batch_strs)
            print(batch_tokens.size())
            print(masked_batch_tokens.size())
            print(results["logits"].size())
            print(pred.size())
            print(target.size())
            print(f"At Epoch: %.1f"% i)
            print(f"Loss %.4f"% loss)
            elapsed = time.time() - start_time
            print(f"time elapsed %.4f"% elapsed)
            torch.save(model.state_dict(), "../data/esm_t12_85M_UR50S_0pt25_vp1s_20211026.pt")

            torch.save(model.state_dict(), "../data/esm_t12_85M_UR50S_0pt25_vp1s_20211026.pt")