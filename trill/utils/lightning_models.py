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
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, T5EncoderModel, T5Tokenizer, AutoModelForSeq2SeqLM
from esm.inverse_folding.multichain_util import sample_sequence_in_complex
# from colossalai.nn.optimizer import HybridAdam, CPUAdam
from deepspeed.ops.adam import FusedAdam
from tqdm import tqdm
import re
from pytorch_lightning.callbacks import BasePredictionWriter
import torch
from torch.utils.data import Dataset

ESM_ALLOWED_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

class ESM(pl.LightningModule):
    def __init__(self, model, lr, args):
        super().__init__()
        self.esm, self.alphabet = model
        self.repr_layers = [(i + self.esm.num_layers + 1) % (self.esm.num_layers + 1) for i in [-1]]
        self.reps = []
        self.lr = lr
        if args.command == 'finetune':
            self.strat = args.strategy
        else:
            self.strat = None
        self.sample_seqs = []
 

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
        if 'offload' in str(self.strat):
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
            # print("*** CPU offloading can't currently be used with TRILL and ProtGPT2 ***")
            # raise RuntimeError
            optimizer = DeepSpeedCPUAdam(self.model.parameters(), lr=self.lr)
        elif 'fsdp' in self.strat:
            print("*** FSDP can't currently be used with TRILL and ProtGPT2 ***")
            raise RuntimeError
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
        if len(seqs) == 1:
            modded_seqs = ''
            for seq in seqs[0]:
                modded_seqs += seq
                modded_seqs += ' '
            modded_seqs = (modded_seqs[:-1],)
        else:
            modded_seqs = []
            for seq in seqs:
                temp_prot = ''
                for aa in seq:
                    temp_prot += aa
                    temp_prot += ' '
                modded_seqs.append(temp_prot)

        token_encoding = self.tokenizer.batch_encode_plus(modded_seqs, 
                add_special_tokens=True, padding='longest')
        input_ids = torch.tensor(token_encoding['input_ids'])
        attention_mask = torch.tensor(token_encoding['attention_mask'])

        if next(self.model.parameters()).is_cuda:
            embedding_repr = self.model(input_ids.cuda(), attention_mask=attention_mask.cuda())
        else:
            embedding_repr = self.model(input_ids, attention_mask=attention_mask)
        emb = embedding_repr.last_hidden_state.squeeze(0)
        if len(seqs) == 1:
            protein_emb = emb.mean(dim=0)
            return tuple((protein_emb, label[0]))
        else:
            protein_emb = emb.mean(dim=1)
            return list(zip(protein_emb, label))


    

class ZymCTRL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if int(args.GPUs) == 1:
            device_map = {'transformer.wte': 0, 'lm_head': 0, 'transformer.wpe': 0, 'transformer.drop': 0, 'transformer.h.0': 0, 'transformer.h.1': 0, 'transformer.h.2': 0, 'transformer.h.3': 0, 'transformer.h.4': 0, 'transformer.h.5': 0, 'transformer.h.6': 0, 'transformer.h.7': 0, 'transformer.h.8': 0, 'transformer.h.9': 0, 'transformer.h.10': 0, 'transformer.h.11': 0, 'transformer.h.12': 0, 'transformer.h.13': 0, 'transformer.h.14': 0, 'transformer.h.15': 0, 'transformer.h.16': 0, 'transformer.h.17': 0, 'transformer.h.18': 0, 'transformer.h.19': 0, 'transformer.h.20': 0, 'transformer.h.21': 0, 'transformer.h.22': 0, 'transformer.h.23': 0, 'transformer.h.24': 0, 'transformer.h.25': 0, 'transformer.h.26': 0, 'transformer.h.27': 0, 'transformer.h.28': 0, 'transformer.h.29': 0, 'transformer.h.30': 0, 'transformer.h.31': 0, 'transformer.h.32': 0, 'transformer.h.33': 0, 'transformer.h.34': 0, 'transformer.h.35': 0, 'transformer.ln_f': 0}
            self.model = AutoModelForCausalLM.from_pretrained("nferruz/ZymCTRL", device_map=device_map)
        elif int(args.GPUs) > 1 and args.command == 'lang_gen':
            self.model = AutoModelForCausalLM.from_pretrained("nferruz/ZymCTRL", device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained("nferruz/ZymCTRL", low_cpu_mem_usage=True)

        self.tokenizer = AutoTokenizer.from_pretrained("nferruz/ZymCTRL")
        self.special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']
        if 'lr' in args:
            self.ctrl_tag = str(args.ctrl_tag)
            self.lr = float(args.lr)
            self.strat = str(args.strategy)
    
    def training_step(self, batch, batch_idx):
        sequence = batch["Labels"][0]
        separator = '<sep>'
        control_tag_length = len(self.tokenizer(self.ctrl_tag+separator)['input_ids'])
        available_space = 1021 - control_tag_length
        if len(sequence) > available_space:
            total_length = control_tag_length + len(sequence[:available_space]) + 1
            seq = f"{self.ctrl_tag}{separator}{sequence[:available_space]}<|endoftext|>"
        else:
            total_length = control_tag_length + len(sequence) + 3
            seq = f"{self.ctrl_tag}{separator}<start>{sequence}<end><|endoftext|>"


        tokenized = self.tokenizer(
        seq,
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
            optimizer = DeepSpeedCPUAdam(self.model.parameters(), lr=self.lr)
        elif 'fsdp' in self.strat:
            print("*** FSDP can't currently be used with TRILL and ProtGPT2 ***")
            raise RuntimeError
            # optimizer = torch.optim.Adam(self.trainer.model.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def generator(self, tag, seed_seq = "", max_length = 100, do_sample = True, temperature = 1.0, top_k = 9, repetition_penalty = 1.2, num_return_sequences = 1, eos_token_id=1, pad_token_id=0, device = 'cpu'):
        tokenized_tag = self.tokenizer.encode(tag, return_tensors='pt').to(device)
        out = self.model.generate(tokenized_tag.to(device), top_k=top_k, repetition_penalty=repetition_penalty,max_length=max_length,eos_token_id=1,pad_token_id=0,do_sample=do_sample,num_return_sequences=1)
        out = out.squeeze(0)
        ppls = [(self.tokenizer.decode(output), self.calculatePerplexity(output)) for output in out.unsqueeze(0)]
        
        sequence_ppl = [(self.remove_characters(x[0], self.special_tokens), x[1]) for x in ppls]
        return sequence_ppl

    
    def calculatePerplexity(self, input_ids):
        "This function computes perplexities for the generated sequences. Got this from https://huggingface.co/nferruz/ZymCTRL"
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        return math.exp(loss)
    
    def remove_characters(self, sequence, char_list):
        "This function removes special tokens used during training. Got this from https://huggingface.co/nferruz/ZymCTRL"
        columns = sequence.split('<sep>')
        seq = columns[1]
        for char in char_list:
            seq = seq.replace(char, '')
        return seq
    
class ProstT5(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.command = args.command
        if self.command == 'embed':
            if int(args.GPUs) > 1:
                self.model = T5EncoderModel.from_pretrained("Rostlab/ProstT5", device_map="auto")
            else:
                self.model = T5EncoderModel.from_pretrained("Rostlab/ProstT5", low_cpu_mem_usage=True)
        elif self.command == 'fold' or self.command == 'inv_fold_gen':
            if int(args.GPUs) > 1:
                self.model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/ProstT5", device_map="auto")
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/ProstT5", low_cpu_mem_usage=True)
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
        if int(args.GPUs) >= 1:
            self.model = self.model.half()
    
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
            # print("*** CPU offloading can't currently be used with TRILL and ProtGPT2 ***")
            # raise RuntimeError
            optimizer = DeepSpeedCPUAdam(self.model.parameters(), lr=self.lr)
        elif 'fsdp' in self.strat:
            print("*** FSDP can't currently be used with TRILL and ProtGPT2 ***")
            raise RuntimeError
            # optimizer = torch.optim.Adam(self.trainer.model.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def predict_step(self, batch, batch_idx):
        if self.command== 'embed':
            label, _ = batch
            seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in batch[1]]
            seqs = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s
                        for s in seqs
                        ]
            ids = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest",return_tensors='pt')
            if next(self.model.parameters()).is_cuda:
                embedding_repr = self.model(ids.input_ids.cuda(), attention_mask=ids.attention_mask.cuda())
            else:
                embedding_repr = self.model(ids.input_ids, attention_mask=ids.attention_mask)
            emb = embedding_repr.last_hidden_state.squeeze(0)
            if len(seqs) == 1:
                protein_emb = emb.mean(dim=0)
                return tuple((protein_emb, label[0]))
            else:
                protein_emb = emb.mean(dim=1)
                return list(zip(protein_emb, label))
        elif self.command == 'fold':
            label, _ = batch
            seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in batch[1]]
            seqs = [ "<AA2fold>" + " " + s for s in seqs]
            ids = self.tokenizer.batch_encode_plus(seqs,
                                  add_special_tokens=True,
                                  padding="longest",
                                  return_tensors='pt')
            gen_kwargs_aa2fold = {
                  "do_sample": True,
                  "num_beams": 3, 
                  "top_p" : 0.95, 
                  "temperature" : 1.2, 
                  "top_k" : 6,
                  "repetition_penalty" : 1.2,
            }
            if next(self.model.parameters()).is_cuda:
                translations = self.model.generate( 
                ids.input_ids.cuda(), 
                attention_mask=ids.attention_mask.cuda(), 
                max_length=max([ len(s) for s in seqs]),
                min_length=min([ len(s) for s in seqs]), 
                early_stopping=True,
                num_return_sequences=1,
                **gen_kwargs_aa2fold
                )
            else:
                translations = self.model.generate( 
                ids.input_ids, 
                attention_mask=ids.attention_mask, 
                max_length=max([ len(s) for s in seqs]),
                min_length=min([ len(s) for s in seqs]), 
                early_stopping=True,
                num_return_sequences=1,
                **gen_kwargs_aa2fold
                )
            decoded_translations = self.tokenizer.batch_decode(translations, skip_special_tokens=True)
            structure_sequences = [ "".join(ts.split(" ")) for ts in decoded_translations ]
            if len(seqs) == 1:
                return tuple((structure_sequences, label[0]))
            else:
                return list(zip(structure_sequences, label))
            
        elif self.command == 'inv_fold_gen':
            label, seq = batch
            seq = seq[0].lower()
            sequence_examples_backtranslation = [ "<fold2AA>" + " " + s for s in seq]
            ids_backtranslation = self.tokenizer.batch_encode_plus(sequence_examples_backtranslation,
                                  add_special_tokens=True,
                                  padding="longest",
                                  return_tensors='pt')
            
            gen_kwargs_fold2AA = {
            "do_sample": True,
            "top_p" : 0.90,
            "temperature" : 1.1,
            "top_k" : 6,
            "repetition_penalty" : 1.2,
            }
            if next(self.model.parameters()).is_cuda:
                backtranslations = self.model.generate( 
                ids_backtranslation.input_ids.cuda(), 
                attention_mask=ids_backtranslation.attention_mask.cuda(), 
                max_length=max([ len(s) for s in seq]),
                min_length=min([ len(s) for s in seq]), 
                early_stopping=True,
                num_return_sequences=1,
                **gen_kwargs_fold2AA
                )
            else:
                backtranslations = self.model.generate( 
                ids_backtranslation.input_ids, 
                attention_mask=ids_backtranslation.attention_mask, 
                max_length=max([ len(s) for s in seq]),
                min_length=min([ len(s) for s in seq]), 
                early_stopping=True,
                num_return_sequences=1,
                **gen_kwargs_fold2AA
                )

            decoded_backtranslations = self.tokenizer.batch_decode(backtranslations, skip_special_tokens=True)
            aminoAcid_sequences = "".join(decoded_backtranslations)
            return aminoAcid_sequences

class CustomWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))


class Custom3DiDataset(Dataset):
    def __init__(self, file_path):
        self.sequences = []
        self.labels = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    self.labels.append(line.strip())
                else:
                    self.sequences.append(line.strip())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        return label, sequence
