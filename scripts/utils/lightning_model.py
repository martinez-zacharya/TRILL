import pytorch_lightning as pl
import torch
import esm
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sys
sys.path.insert(0, 'utils')
from mask import maskInputs
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from esm.inverse_folding.multichain_util import sample_sequence_in_complex

class LitModel(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.esm, self.alphabet = model
        self.repr_layers = [(i + self.esm.num_layers + 1) % (self.esm.num_layers + 1) for i in [-1]]
        self.reps = []
        self.lr = lr
        self.sample_seqs = []

    def training_step(self, batch, batch_idx):
        labels, seqs, toks = batch
        del labels, seqs
        masked_toks = maskInputs(toks)
        output = self.esm(masked_toks, repr_layers = [-1], return_contacts=False)
        loss = F.cross_entropy(output['logits'].permute(0,2,1), toks)
        self.log("loss", loss)
        return {"loss": loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.esm.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
    
    def predict_step(self, batch, batch_idx):
        labels, seqs, toks = batch
        pred = self.esm(toks, repr_layers=self.repr_layers, return_contacts=False)
        representations = {layer: t.to(device="cpu") for layer, t in pred["representations"].items()}
        rep_numpy = representations[self.repr_layers[0]].cpu().detach().numpy()
        for i in range(len(rep_numpy)):
            self.reps.append(tuple([rep_numpy[i].mean(0), labels[i]]))
        return True
        
    

class coordDataset(torch.utils.data.Dataset):
    def __init__(self, input):
        self.input = input
    def __getitem__(self, idx):
        coords, seq = self.input[idx]
        return coords, seq
    def __len__(self):
        return len(self.input)
        
    
    
    