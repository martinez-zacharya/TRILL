# From https://github.com/beckham-lab/EpHod/blob/master/ephod/models.py

"""Modular functions/classes for building and running EpHod model
Author: Japheth Gado
"""




import subprocess
import os
import builtins
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn.parallel import DataParallel
import esm
import logging
import json
import requests
from collections import OrderedDict








def print(*args, **kwargs):
    '''Custom print function to always flush output when verbose'''

    builtins.print(*args, **kwargs, flush=True)
    
    

def replace_noncanonical(seq, replace_char='X'):
    '''Replace all non-canonical amino acids with a specific character'''

    for char in ['B', 'J', 'O', 'U', 'Z']:
        seq = seq.replace(char, replace_char)
    return seq


def read_fasta(fasta, return_as_dict=False):
    '''Read the protein sequences in a fasta file. If return_as_dict, return a dictionary
    with headers as keys and sequences as values, else return a tuple, 
    (list_of_headers, list_of_sequences)'''
    
    headers, sequences = [], []

    with open(fasta, 'r') as fast:
        
        for line in fast:
            if line.startswith('>'):
                head = line.replace('>','').strip()
                headers.append(head)
                sequences.append('')
            else :
                seq = line.strip()
                if len(seq) > 0:
                    sequences[-1] += seq

    if return_as_dict:
        return dict(zip(headers, sequences))
    else:
        return (headers, sequences)



def download_models(get_from='zenodo'):
    '''Download saved models (EpHod and AAC-SVR)'''
    
    if get_from == 'googledrive':
        
        # Download from Google drive
        glink = "https://drive.google.com/drive/folders/138cnx4hFrzNODGK6A_yd9wo7WupKpSjI?usp=share_link/"
        cmd = f"gdown --folder {glink}"
        print('Downloading EpHod models from Google drive with gdown\n')
        _ = subprocess.call(cmd, shell=True) # Download model from google drive 

    elif get_from == 'zenodo':
        
        # Download from Zenodo
        zlink = "https://zenodo.org/record/8011249/files/saved_models.tar.gz?download=1"
        print('Downloading EpHod models from Zenodo with requests\n')
        response = requests.get(zlink, stream=True)
        if response.status_code == 200:
            with requests.get(zlink, stream=True) as r:
                r.raise_for_status()
                with open("saved_models.tar.gz", 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
        else:
            print(f"Request failed with status code {response.status_code}")

    
    else: 
        raise ValueError(f"Value of get_from ({get_from}) must be 'googledrive' or 'zenodo'")
        
    
    # Move downloaded models to proper location
    this_dir, this_filename = os.path.split(__file__)
    if get_from == 'zenodo':
        # Untar downloaded file
        tarfile = os.path.join(this_dir, 'saved_models') 
        _ = subprocess.call(f"tar -xvzf {tarfile}", shell=True)
        _ = subprocess.call(f"rm -rfv {tarfile}", shell=True)
    
    save_path = os.path.join(this_dir, 'saved_models') 
    cmd = f"mv -f ./saved_models {save_path}/"
    print(cmd)
    print(f'\nMoving downloaded models to {save_path}')
    _ = subprocess.call(cmd, shell=True)
    error_msg = "RLAT model failed to download!"
    assert os.path.exists(f"{save_path}/RLAT/RLAT.pt"), error_msg
    
    
    
    
    
    
    
    
def torchActivation(activation='elu'):
    '''Return an activation function from torch.nn'''

    if activation == 'relu':
        return nn.ReLU()

    elif activation == 'leaky_relu':
        return nn.LeakyReLU()

    elif activation == 'elu':
        return nn.ELU()

    elif activation == 'selu':
        return nn.SELU()
    
    elif activation == 'gelu':
        return nn.GELU()
        

def read_json(path):
    '''Return a dictionry read from a json file'''
    
    f = open(path, 'r')
    readdict = json.load(f)
    f.close()
    
    return readdict 

class CustomDataset(Dataset):
    def __init__(self, reps, masks):
        self.reps = reps
        self.masks = masks

    def __len__(self):
        return len(self.reps)

    def __getitem__(self, idx):
        return self.reps[idx], self.masks[idx]



def count_parameters(model):
    '''Return a count of parameters and tensor shape of PyTorch model''' 
    
    counted = {}
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = param.numel()
            total += count
            counted[name] = count
    counted['FULL_MODEL'] = total

    return counted
    







class ResidualDense(nn.Module):
    '''A single dense layer with residual connection'''
    
    def __init__(self, dim=2560, dropout=0.1, activation='elu', random_seed=0):
        
        super(ResidualDense, self).__init__()
        _ = torch.manual_seed(random_seed)
        self.dense = nn.Linear(dim, dim)
        self.batchnorm = nn.BatchNorm1d(dim)
        self.activation = torchActivation(activation)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        x0 = x
        x = self.dense(x)
        x = self.batchnorm(x)
        x = self.activation(x)        
        x = self.dropout(x)
        x = x0 + x
        
        return x








class LightAttention(nn.Module):
    '''Convolution model with attention to learn pooled representations from embeddings'''

    def __init__(self, dim=1280, kernel_size=7, random_seed=0):
        
        super(LightAttention, self).__init__()
        _ = torch.manual_seed(random_seed)        
        samepad = kernel_size // 2
        self.values_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=samepad)
        self.weights_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=samepad)
        self.softmax = nn.Softmax(dim=-1)
    
    
    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[2], dtype=torch.int32)  # Don't mask out
        values = self.values_conv(x)
        fill_value = -1e6 if values.dtype == torch.float32 else -65504
        values = values.masked_fill(mask[:,None,:]==0, fill_value)
        weights = self.weights_conv(x)
        weights = weights.masked_fill(mask[:,None,:]==0, fill_value)
        weights = self.softmax(weights)
        x_sum = torch.sum(values * weights, dim=-1) # Attention-weighted pooling
        x_max, _ = torch.max(values, dim=-1) # Max pooling
        x = torch.cat([x_sum, x_max], dim=1)
        
        return x, weights
    
    
    





class ResidualLightAttention(pl.LightningModule):
    '''Model consisting of light attention followed by residual dense layers'''
    
    def __init__(self, dim=1280, kernel_size=9, dropout=0.5,
                 activation='relu', res_blocks=4, random_seed=0):

        super(ResidualLightAttention, self).__init__()
        torch.manual_seed(random_seed)
        self.light_attention = LightAttention(dim, kernel_size, random_seed)
        self.batchnorm = nn.BatchNorm1d(2 * dim)                
        self.dropout = nn.Dropout(dropout)        
        self.residual_dense = nn.ModuleList()        
        for i in range(res_blocks):
            self.residual_dense.append(
                ResidualDense(2 * dim, dropout, activation, random_seed)
                )
        self.output = nn.Linear(2 * dim, 1)
        
        
    def forward(self, x, mask=None):
        x, mask = x
        x = x.unsqueeze(0)
        mask = mask.unsqueeze(0)
        x_2, weights = self.light_attention(x, mask)
        x_2 = self.batchnorm(x_2)
        x_2 = self.dropout(x_2)
        for layer in self.residual_dense:
            x_2 = layer(x_2)
        y = self.output(x_2).flatten()

        return y, x_2, weights
    
    #Not real training function
    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        labels, seqs, toks = batch
        del labels, seqs, batch_idx
        output = self.esm(toks, repr_layers = [-1], return_contacts=False)
        loss = torch.nn.functional.cross_entropy(output['logits'].permute(0,2,1), toks)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.light_attention.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


class ESM1v(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.esm, self.alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
        self.repr_layers = [(i + self.esm.num_layers + 1) % (self.esm.num_layers + 1) for i in [-1]]
 

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        labels, seqs, toks = batch
        del labels, seqs, batch_idx
        masked_toks = maskInputs(toks, self.esm)
        output = self.esm(masked_toks, repr_layers = [-1], return_contacts=False)
        loss = F.cross_entropy(output['logits'].permute(0,2,1), toks)
        self.log("loss", loss)
        del masked_toks, toks
        return {"loss": loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.esm.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def predict_step(self, batch, batch_idx):
        toks = batch
        toks = toks.unsqueeze(0)
        pred = self.esm(toks, repr_layers=self.repr_layers, return_contacts=False)
        representations = {layer: t.to(device="cpu") for layer, t in pred["representations"].items()}
        rep_numpy = representations[33].cpu().detach().numpy()
        # rep_numpy.transpose(2,1)
        # for i in range(len(rep_numpy)):
            # self.reps.append(tuple([rep_numpy[i].mean(0), labels[i]]))
            # reps.append(tuple([rep_numpy[i].mean(0), labels[i]]))
        # newdf = pd.DataFrame(reps, columns = ['Embeddings', 'Label'])
        # finaldf = newdf['Embeddings'].apply(pd.Series)
        # finaldf['Label'] = newdf['Label']
        # return finaldf
        return rep_numpy


class EpHodModel():
    def __init__(self, args):
        self.esm1v_model, self.esm1v_batch_converter, self.pl = self.load_ESM1v_model()
        self.rlat_model = self.load_RLAT_model()
        _ = self.esm1v_model.eval()
        _ = self.rlat_model.eval()
        if int(args.GPUs) >= 1:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
    
    def load_ESM1v_model(self):
        '''Return pretrained ESM1v model weights and batch converter'''
        model = ESM1v()
        batch_converter = model.alphabet.get_batch_converter()
        
        return model.esm, batch_converter, model
    
    
    def get_ESM1v_embeddings(self, accs, seqs, args):
        '''Return per-residue embeddings (padded) for protein sequences from ESM1v model'''

        seqs = [replace_noncanonical(seq, 'X') for seq in seqs]

        data = [(accs[i], seqs[i]) for i in range(len(accs))]
        batch_labels, batch_strs, batch_tokens = self.esm1v_batch_converter(data)
        batch_tokens = batch_tokens.to(device=self.device, non_blocking=True)


        if int(args.GPUs) == 0:
            trainer = pl.Trainer(enable_checkpointing=False, num_nodes=int(args.nodes), enable_progress_bar=False)
        else:
            trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), accelerator='gpu', num_nodes=int(args.nodes), precision = 16,  enable_progress_bar=False)
        # self.esm1v_model = self.esm1v_model.to(device=self.device)
        logging.getLogger("lightning").addHandler(logging.NullHandler())
        logging.getLogger("lightning").propagate = False
        reps = trainer.predict(self.pl, batch_tokens)
        # emb = self.esm1v_model(batch_tokens, repr_layers=[33], return_contacts=False)
        # emb = emb["representations"][33]
        # emb = emb.transpose(2,1) # From (batch, seqlen, features) to (batch, features, seqlen)
        # emb = emb.to(self.device)


        return reps
    
    
    def load_RLAT_model(self):
        '''Return fine-tuned residual light attention top model'''

        # Path to RLAT model
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".trill_cache")

        params_path = os.path.join(cache_dir, 'EpHod_Models', 'RLAT', 'params.json')
        rlat_path = os.path.join(cache_dir, 'EpHod_Models', 'RLAT', 'RLAT.pt')
        
        # Load RLAT model from path
        checkpoint = torch.load(rlat_path)
        params = read_json(params_path)        
        model = ResidualLightAttention(**params)
        # model = DataParallel(model)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)

        return model

    
    def batch_predict(self, accs, seqs, args):
        '''Predict pHopt with EpHod on a batch of sequences'''
        
        reps = self.get_ESM1v_embeddings(accs, seqs, args)
        maxlen = reps[0].shape[-2]
        masks = [[1] * len(seqs[i]) + [0] * (maxlen - len(seqs[i])) \
                 for i in range(len(seqs))]
        masks = torch.tensor(masks, dtype=torch.int32)
        masks = masks.to(self.device)
        reps = torch.from_numpy(reps[0])
        reps = reps.permute(0, 2, 1)
        logging.getLogger("lightning").addHandler(logging.NullHandler())
        logging.getLogger("lightning").propagate = False
        if int(args.GPUs) == 0:
            trainer = pl.Trainer(enable_checkpointing=False, num_nodes=int(args.nodes), enable_progress_bar=False)
        else:
            trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), accelerator='gpu', num_nodes=int(args.nodes), precision = 16, enable_progress_bar=False)

        dataset = CustomDataset(reps, masks)
        output = trainer.predict(self.rlat_model, dataset)
        return output[0]
    
    
    

