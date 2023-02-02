import pytorch_lightning as pl
import torch
import esm
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sys
import gc
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# sys.path.insert(0, 'utils')
from utils.mask import maskInputs
from utils.update_weights import weights_update
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from esm.inverse_folding.multichain_util import sample_sequence_in_complex
# from colossalai.nn.optimizer import HybridAdam, CPUAdam
from deepspeed.ops.adam import FusedAdam


class ESM(pl.LightningModule):
    def __init__(self, model, lr, leggo = False):
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
        reps = []
        for i in range(len(rep_numpy)):
            # self.reps.append(tuple([rep_numpy[i].mean(0), labels[i]]))
            reps.append(tuple([rep_numpy[i].mean(0), labels[i]]))
        # newdf = pd.DataFrame(reps, columns = ['Embeddings', 'Label'])
        # finaldf = newdf['Embeddings'].apply(pd.Series)
        # finaldf['Label'] = newdf['Label']
        # return finaldf
        return reps
    
    
class ProtGPT2(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
        self.model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2")
        self.lr = lr
        # if pretrained != None:
        #     self.model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2")
        #     # self.model = self.model.load_from_checkpoint(pretrained)
        #     # self.model = weights_update(self.model, checkpoint=pretrained)
        # else:
        #     self.model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2")

    
    def training_step(self, batch, batch_idx):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized = self.tokenizer(
        batch["Labels"],
        padding=True,
        return_special_tokens_mask=True
    )
        att = torch.LongTensor(tokenized['attention_mask']).cuda()
        data_collator = DataCollatorForLanguageModeling(tokenizer = self.tokenizer, mlm=False)
        collated = data_collator([tokenized['input_ids']])
        outputs = self.model(collated['input_ids'].cuda(), labels = collated['labels'].cuda(), attention_mask = att, return_dict = True)
        loss = outputs[0]
        self.log("loss", loss)
        return(loss)
        
    def configure_optimizers(self):
        optimizer = DeepSpeedCPUAdam(self.model.parameters(), lr=1e-5)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        # optimizer = FusedAdam(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def generate(self, seed_seq = "M", max_length = 100, do_sample = True, top_k = 950, repetition_penalty = 1.2, num_return_sequences = 5, eos_token_id=0):
        generator = pipeline('text-generation', model = self.model, tokenizer=self.tokenizer)
        outseqs = generator(seed_seq, max_length=max_length, do_sample =do_sample, top_k=top_k, repetition_penalty=repetition_penalty, num_return_sequences=num_return_sequences, eos_token_id=eos_token_id)
        outseqs = [samp['generated_text'].replace('\n','') for samp in outseqs]
        return outseqs

    
        
    
    
from pytorch_lightning.callbacks import BasePredictionWriter

class CustomWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))