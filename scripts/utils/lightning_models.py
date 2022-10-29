import pytorch_lightning as pl
import torch
import esm
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sys
import gc
sys.path.insert(0, 'utils')
from mask import maskInputs
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from esm.inverse_folding.multichain_util import sample_sequence_in_complex

class ESM(pl.LightningModule):
    def __init__(self, model, lr, leggo):
        super().__init__()
        self.esm, self.alphabet = model
        self.repr_layers = [(i + self.esm.num_layers + 1) % (self.esm.num_layers + 1) for i in [-1]]
        self.reps = []
        self.lr = lr
        self.sample_seqs = []
        if leggo:
            self.leggo = True
        else:
            self.leggo = False

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        labels, seqs, toks = batch
        del labels, seqs, batch_idx
        masked_toks = maskInputs(toks)
        output = self.esm(masked_toks, repr_layers = [-1], return_contacts=False)
        loss = F.cross_entropy(output['logits'].permute(0,2,1), toks)
        self.log("loss", loss)
        del masked_toks, toks
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass  
        return {"loss": loss}
    
    def configure_optimizers(self):
        if self.leggo:
            optimizer = DeepSpeedCPUAdam(self.esm.parameters(), lr=self.lr)
            return optimizer
        else:
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
        
        
class ProtGPT2(pl.LightningModule):
    def __init__(self, pretrained = None):
        super().__init__()
        if pretrained != None:
            self.model = AutoModelForCausalLM.from_pretrained(pretrained)
        else:
            self.model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2")
        self.tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")

    def training_step(self, batch, batch_idx):
        # if len(batch) == 1:
        #     batch = batch[0]
        # tokenizer_output = self.tokenizer(batch)
        print(batch.input_ids)
        # data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        # collated_data = data_collator(tokenizer_output)
        outputs = self.model(batch)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
    
    def predict_step(self, seed_seq="M", max_length=333, do_sample = True, top_k=950, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=0):
        generator = pipeline('text-generation', model = self.model, tokenizer=self.tokenizer)
        outseqs = generator(seed_seq, max_length, do_sample, top_k, repetition_penalty, num_return_sequences, eos_token_id)
        return outseqs

class coordDataset(torch.utils.data.Dataset):
    def __init__(self, input):
        self.input = input
    def __getitem__(self, idx):
        coords, seq = self.input[idx]
        return coords, seq
    def __len__(self):
        return len(self.input)
    
class ProtGPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, input):
        self.labels = list(input.keys())
        self.seqs = list(input.values())
    def __getitem__(self, idx):
        label = self.labels[idx]
        seq = self.seqs[idx]
        return {'input_ids': seq, 'text': label }
    def __len__(self):
        return len(self.labels)
        
    
    
    