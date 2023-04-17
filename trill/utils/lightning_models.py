import pytorch_lightning as pl
import torch
import esm
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sys
import gc
import os
import math
from tqdm import trange
import random
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# sys.path.insert(0, 'utils')
from utils.mask import maskInputs
from utils.update_weights import weights_update
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, T5EncoderModel, T5Tokenizer
from esm.inverse_folding.multichain_util import sample_sequence_in_complex
# from colossalai.nn.optimizer import HybridAdam, CPUAdam
from deepspeed.ops.adam import FusedAdam
from tqdm import tqdm

ESM_ALLOWED_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

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
        masked_toks = maskInputs(toks, self.esm)
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
    
class ESM_Gibbs_old(pl.LightningModule):
    def __init__(self, model, lr, leggo = False, seed='M', total=1, max_len=100, temp = 1, top_k = 950):
        super().__init__()
        self.esm, self.alphabet = model
        self.repr_layers = [(i + self.esm.num_layers + 1) % (self.esm.num_layers + 1) for i in [-1]]
        self.reps = []
        self.lr = lr
        self.sample_seqs = []
        self.seed = seed
        self.total = total
        self.max_len = max_len
        self.temp = temp
        self.top_k = top_k
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
    
    def step(self, out, idx):
        flag = False
        while flag == False:
            logits = out[idx]
            logits = logits / self.temp
            valid_idx = list(range(len(logits)))
            if self.top_k <= 0 or self.top_k > len(logits[valid_idx]):
                kth_vals, kth_idx = torch.topk(logits, len(logits[valid_idx]))
            else:
                kth_vals, kth_idx = torch.topk(logits, self.top_k)
            dist = torch.distributions.categorical.Categorical(logits=kth_vals)
            new_idx = kth_idx[dist.sample()]
            if valid_idx[new_idx] != 32 and valid_idx[new_idx] != 1 and valid_idx[new_idx] != 3 and valid_idx[new_idx] != 0 and valid_idx[new_idx] != 2 and valid_idx[new_idx] != 31 and valid_idx[new_idx] != 30 and valid_idx[new_idx] != 29:
                flag = True
        return torch.tensor(valid_idx[new_idx])

    def untokenize(self, batch, alphabet):
        out = [ "".join([alphabet.get_tok(seq[i]) for i in range(0, len(seq))]) for seq in batch]
        return out

    def configure_optimizers(self):
        if self.leggo:
            optimizer = DeepSpeedCPUAdam(self.esm.parameters(), lr=self.lr)
            return optimizer
        else:
            optimizer = torch.optim.Adam(self.esm.parameters(), lr=self.lr)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]
    
    def predict(self):
        batch_converter = self.alphabet.get_batch_converter()
        seed = self.seed
        sequence_to_add_to = seed
        for j in range(self.max_len):
            prev_seq = sequence_to_add_to + "<mask>"
            new_input = [(seed, prev_seq)]

            label, seq, toks = batch_converter(new_input)
            self.esm = self.esm.cuda()
            out = self.esm(toks.cuda())['logits'].squeeze(0)
            toks = toks.cpu().detach()
            masks = np.where(toks==32)[1]
            for index in masks:
                idx = self.step(out, index)
                toks[0][index] = idx
                newseq = self.untokenize(toks, self.alphabet)
                sequence_to_add_to+=newseq[0][-6]
            torch.cuda.empty_cache()
            del label, seq, toks, out
        return(sequence_to_add_to)
    
    
class ProtGPT2(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if int(args.GPUs) == 1:
            device_map = {'transformer.wte': 0, 'lm_head': 0, 'transformer.wpe': 0, 'transformer.drop': 0, 'transformer.h.0': 0, 'transformer.h.1': 0, 'transformer.h.2': 0, 'transformer.h.3': 0, 'transformer.h.4': 0, 'transformer.h.5': 0, 'transformer.h.6': 0, 'transformer.h.7': 0, 'transformer.h.8': 0, 'transformer.h.9': 0, 'transformer.h.10': 0, 'transformer.h.11': 0, 'transformer.h.12': 0, 'transformer.h.13': 0, 'transformer.h.14': 0, 'transformer.h.15': 0, 'transformer.h.16': 0, 'transformer.h.17': 0, 'transformer.h.18': 0, 'transformer.h.19': 0, 'transformer.h.20': 0, 'transformer.h.21': 0, 'transformer.h.22': 0, 'transformer.h.23': 0, 'transformer.h.24': 0, 'transformer.h.25': 0, 'transformer.h.26': 0, 'transformer.h.27': 0, 'transformer.h.28': 0, 'transformer.h.29': 0, 'transformer.h.30': 0, 'transformer.h.31': 0, 'transformer.h.32': 0, 'transformer.h.33': 0, 'transformer.h.34': 0, 'transformer.h.35': 0, 'transformer.ln_f': 0}
            self.model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2", device_map=device_map)
        elif int(args.GPUs) > 1 and args.command == 'lang_gen':
            self.model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2", device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2", low_cpu_mem_usage=True)

        self.tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
        if 'lr' in args:
            self.lr = float(args.lr)
            self.strat = str(args.strategy)
    
    def training_step(self, batch, batch_idx):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized = self.tokenizer(
        batch["Labels"],
        padding=True,
        return_special_tokens_mask=True
    )
        if next(self.model.parameters()).is_cuda:
            att = torch.LongTensor(tokenized['attention_mask']).cuda()
        else:
            att = torch.LongTensor(tokenized['attention_mask'])
        data_collator = DataCollatorForLanguageModeling(tokenizer = self.tokenizer, mlm=False)
        collated = data_collator([tokenized['input_ids']])
        if next(self.model.parameters()).is_cuda:
            outputs = self.model(collated['input_ids'].cuda(), labels = collated['labels'].cuda(), attention_mask = att, return_dict = True)
        else:
            outputs = self.model(collated['input_ids'], labels = collated['labels'], attention_mask = att, return_dict = True)
        loss = outputs[0]
        self.log("loss", loss)
        return(loss)
        
    def configure_optimizers(self):
        if 'offload' in self.strat:
            print("*** CPU offloading can't currently be used with TRILL and ProtGPT2 ***")
            raise RuntimeError
            # optimizer = DeepSpeedCPUAdam(self.model.parameters(), lr=self.lr)
        elif 'fsdp' in self.strat:
            print("*** FSDP can't currently be used with TRILL and ProtGPT2 ***")
            raise RuntimeError
            return
            # optimizer = torch.optim.Adam(self.trainer.model.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def generate(self, seed_seq = "M", max_length = 100, do_sample = True, temperature = 1.0, top_k = 950, repetition_penalty = 1.2, num_return_sequences = 1, eos_token_id=0):
        generator = pipeline('text-generation', model = self.model, tokenizer=self.tokenizer, device_map='auto')
        outseqs = generator(seed_seq, max_length=max_length, temperature = temperature, do_sample =do_sample, top_k=top_k, repetition_penalty=repetition_penalty, num_return_sequences=num_return_sequences, pad_token_id=eos_token_id, eos_token_id=eos_token_id)
        outseqs = [samp['generated_text'].replace('\n','') for samp in outseqs]
        return outseqs

class ESM_Gibbs(pl.LightningModule):
    """adapted from bert-gen bert-babble.ipynb"""


    def __init__(self, model, args, device="cpu"):
        """
            model should be an object with parameters model, alphabet, and batch_converter
        """
        super().__init__()

        self.model, self.alphabet = model
        self.batch_converter = self.alphabet.get_batch_converter()
        self.GPUs = int(args.GPUs)

        self.valid_aa_idx = sorted([self.alphabet.get_idx(tok) for tok in ESM_ALLOWED_AMINO_ACIDS])

    
    def generate_step(self, out, gen_idx, temperature=None, top_k=0, valid_idx=None):
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

        if(top_k <= 0) or (top_k > len(sub_logits)):
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
    
    def untokenize_batch(self, batch, bos, eos): #TODO: maybe should be moved to the model class, or a model superclass?
        #convert tokens to AAs, but skip the first one, because that one is <cls>
        start_offset = 0
        end_offset = 0
        if bos:
            start_offset = 1
        if eos:
            end_offset = -1
        out = [ "".join([self.alphabet.get_tok(seq[i]) for i in range(0 + start_offset, len(seq) + end_offset) ]) for seq in batch]
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

        labels, strs, tokens = self.batch_converter(batch)
        return tokens

    def generate(self, seed_seq, n_samples = 1, batch_size=1, in_order=True, max_len=None, leader_length=0, leader_length_percent=None, top_k=0, temperature=None, num_iters=10,  burnin=float('inf'),
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

            # cuda = self.cuda
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
                # batch = batch.cuda() if cuda else batch

                indexes, last_i = self.calculate_indexes(indexes, leader_length, max_len, rollover_from_start)

                if num_positions > len(indexes):
                    num_positions = len(indexes)

                flag = True
                while flag == True:
                # for ii in range(num_iters):
                    if 32 not in batch:
                        flag = False
                        break
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
                    if self.GPUs != 0:
                        batch = batch.cuda()
                    out = self.model(batch)["logits"]

                    for batch_index in range(batch_size):
                        for kk in target_indexes[batch_index]:
                            idx = self.generate_step(out[batch_index],
                                                gen_idx=kk,
                                                top_k=top_k,
                                                temperature=temperature,
                                                valid_idx=self.valid_aa_idx)

                            batch[batch_index][kk] = idx

                if batch_n == (n_batches - 1): #last batch, so maybe don't take all of them, just take enough to get to n_samples
                    sequences += self.untokenize_batch(batch, self.alphabet.prepend_bos, self.alphabet.append_eos)[0:n_samples - len(sequences)]
                else:
                    sequences += self.untokenize_batch(batch, self.alphabet.prepend_bos, self.alphabet.append_eos)
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
                batch[batch_index][kk] = self.alphabet.mask_idx

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
        optimizer = torch.optim.Adam(self.esm.parameters(), lr=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


    def log_likelihood(self, seq, with_masking=True, verbose=False, mask_distance=float("inf"), batch_size=None):
        """
            seq: a protein sequence string
            with_masking: if True, then iterate over the sequence masking one position at a time and summing the log likelihoods of the correct choice at the masked positions.
                        if False, then run the model just once, on the unmasked sequence.
            mask_distance: For optimization, when masking individual positions, the distance between masked positions in the same execution, by default only one position is masked per model call.
            batch_size: number of MSAs to run on the gpu at once, if None, then batch_size=len(msa_list). default=None.
        """
        return next(self.log_likelihood_batch([seq], with_masking, verbose, mask_distance, batch_size))

    #TODO: convert to iterator
    def log_likelihood_batch(self, seq_list, with_masking=True, verbose=False, mask_distance=float("inf"), batch_size=None):

        # TODO: Allow batching to calculate likelihoods for multiple sequences at a time (how does padding effect likelihoods for sequences shorter than the longest sequence, hopefully not at all).

        # Inspired by and borrowing code from:
        # https://github.com/facebookresearch/esm/blob/master/variant-prediction/predict.py

        n_batches = len(seq_list)
        if batch_size is None:
            batch_size = n_batches

        reformatted_seq = [(str(idx), self.clean_seed_seq(seq)) for idx, seq in enumerate(seq_list)]
        _, _, tokens = self.batch_converter(reformatted_seq)

        range_start = 1 if self.alphabet.prepend_bos else 0
        end_modifier = -1 if self.alphabet.append_eos else 0

        batch_range_end = [len(seq) + range_start for seq in seq_list]
        overall_range_end = tokens.shape[1] + end_modifier

        assert max(len(seq) for seq in seq_list) == len(range(range_start, overall_range_end))

        for b_idx in range(n_batches):
            assert len(seq_list[b_idx]) == len(range(range_start, batch_range_end[b_idx]))

        # tokens = tokens.cuda() if self.cuda else tokens
        with torch.no_grad():
            if with_masking:
                old_toks = tokens.clone().detach()

                for seq_idx in range(len(reformatted_seq)):
                    likelihood_sum = 0.0
                    likelihood_list = list()

                    original_string = reformatted_seq[seq_idx][1]
                    num_sequences_to_process = int(min(mask_distance, len(original_string)))
                    all_samples_for_seq = [(idx, original_string) for idx, seq in enumerate(range(num_sequences_to_process))]

                    _, _, tokens_for_seq = self.batch_converter(all_samples_for_seq)

                    masked_idx = set()
                    for i_sample in range(num_sequences_to_process):
                        positions = range(range_start + i_sample, batch_range_end[seq_idx], num_sequences_to_process)
                        tokens_for_seq[i_sample, positions] = self.alphabet.mask_idx
                        masked_idx.update(positions)
                    assert len(masked_idx) == len(original_string), sorted(masked_idx)

                    counted_idx = set()
                    for batch_start in range(0, num_sequences_to_process, batch_size):

                        this_batch = tokens_for_seq[batch_start:batch_start + batch_size, :]
                        # this_batch = this_batch.cuda() if self.cuda else this_batch
                        token_probs = torch.log_softmax(self.model(this_batch)['logits'], dim=-1)

                        for i_sample in range(token_probs.shape[0]):
                            for idx_pos in range(range_start, batch_range_end[seq_idx]):
                                if (idx_pos - range_start) % num_sequences_to_process == (i_sample + batch_start):
                                    likelihood = token_probs[i_sample, idx_pos, old_toks[seq_idx, idx_pos].item()]
                                    likelihood_sum += likelihood
                                    likelihood_list.append(likelihood.item())
                                    counted_idx.add(idx_pos)

                    assert len(counted_idx) == len(seq_list[seq_idx]), sorted(counted_idx)

                    yield (float(likelihood_sum / len(seq_list[seq_idx])), likelihood_list)

            else:  # no masking, so we just need to calculate a single forward pass on the unmasked model
                token_probs = torch.log_softmax(self.model(tokens)['logits'], dim=-1)
                for batch_idx in range(n_batches):
                    log_likelihood_sum = 0.0
                    log_likelihood_list = []
                    for idx in range(range_start, batch_range_end[batch_idx]):
                        likelihood = token_probs[batch_idx, idx, tokens[batch_idx, idx].item()]
                        log_likelihood_sum += likelihood
                        log_likelihood_list.append(likelihood.item())
                    yield (float(log_likelihood_sum / len(seq_list[batch_idx])), log_likelihood_list)

    

class ProtT5(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        device_map = {'shared': 'cpu', 'encoder.embed_tokens': 'cpu', 'encoder.block.0': 'cpu', 'encoder.block.1': 'cpu', 'encoder.block.2': 'cpu', 'encoder.block.3': 'cpu', 'encoder.block.4': 'cpu', 'encoder.block.5': 'cpu', 'encoder.block.6': 'cpu', 'encoder.block.7': 'cpu', 'encoder.block.8': 'cpu', 'encoder.block.9': 'cpu', 'encoder.block.10': 'cpu', 'encoder.block.11': 'cpu', 'encoder.block.12': 'cpu', 'encoder.block.13': 'cpu', 'encoder.block.14': 'cpu', 'encoder.block.15': 'cpu', 'encoder.block.16': 'cpu', 'encoder.block.17': 'cpu', 'encoder.block.18': 'cpu', 'encoder.block.19': 'cpu', 'encoder.block.20': 'cpu', 'encoder.block.21': 'cpu', 'encoder.block.22': 'cpu', 'encoder.block.23': 'cpu', 'encoder.final_layer_norm': 'cpu', 'encoder.dropout': 'cpu'}
        if int(args.GPUs) > 0:
            self.model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc')
        else:
            self.model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', device_map=device_map)
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        self.reps = []


    def training_step(self, batch, batch_idx):
        loss = 0
        return {"loss": loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.esm.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
    
    def predict_step(self, batch, batch_idx):
        label, seqs = batch
        modded_seqs = ''
        for seq in seqs[0]:
            modded_seqs += seq
            modded_seqs += ' '
        modded_seqs = (modded_seqs[:-1],)
        token_encoding = self.tokenizer.batch_encode_plus(modded_seqs, 
                add_special_tokens=True, padding='longest')
        input_ids = torch.tensor(token_encoding['input_ids'])
        attention_mask = torch.tensor(token_encoding['attention_mask'])
        if next(self.model.parameters()).is_cuda:
            embedding_repr = self.model(input_ids.cuda(), attention_mask=attention_mask.cuda())
        else:
            embedding_repr = self.model(input_ids, attention_mask=attention_mask)
        emb = embedding_repr.last_hidden_state.squeeze(0)
        protein_emb = emb.mean(dim=0)
        reps = tuple((protein_emb, label[0]))
        # self.reps.append(tuple((protein_emb, label)))
        # reps = []
        # for i in range(len(rep_numpy)):
        #     reps.append(tuple([rep_numpy[i].mean(0), label[i]]))

        return reps


from pytorch_lightning.callbacks import BasePredictionWriter

class CustomWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))
