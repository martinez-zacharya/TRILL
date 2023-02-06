import pytorch_lightning as pl
import torch
import esm
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sys
import gc
import subprocess
import urllib.request
import os
import copy
import gc
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
from colossalai.nn.optimizer import HybridAdam, CPUAdam
from deepspeed.ops.adam import FusedAdam
from colossalai.utils import colo_set_process_memory_fraction
# from utils.protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN, loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB


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
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass
        labels, seqs, toks = batch
        length = len(seqs[0])
        print(labels)
        del labels, seqs, batch_idx
        masked_toks = maskInputs(toks)
        try:
            output = self.esm(masked_toks, repr_layers = [-1], return_contacts=False)
        except Exception as e:
            print(e)
            print(f'Length of input is too long: {length}')
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

class colossal_ESM(pl.LightningModule):
    def __init__(self, cuda_mem_fraction: float = 1.0):
        super().__init__()
        self.cuda_mem_fraction = cuda_mem_fraction
    def configure_sharded_model(self) -> None:
        # create your model here
        self.model, _ = esm.pretrained.esm2_t30_150M_UR50D()
        self.repr_layers = [(i + self.model.num_layers + 1) % (self.model.num_layers + 1) for i in [-1]]


    #     self.esm, self.alphabet = model
    #     self.reps = []
    #     self.lr = lr
    #     self.sample_seqs = []
    #     if leggo:
    #         self.leggo = True
    #     else:
    #         self.leggo = False

    def training_step(self, batch, batch_idx):
        # torch.cuda.empty_cache()
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass
        labels, seqs, toks = batch
        del labels, seqs, batch_idx
        masked_toks = maskInputs(toks)
        output = self.model(masked_toks, repr_layers = [-1], return_contacts=False, colossal = True)
        loss = F.cross_entropy(output['logits'].permute(0,2,1), toks)
        self.log("loss", loss)
        del masked_toks, toks
        return {"loss": loss}
    
    def configure_optimizers(self):
        optimizer = HybridAdam(self.model.parameters(), lr=0.0001)
        return [optimizer]
    
    def predict_step(self, batch, batch_idx):
        labels, seqs, toks = batch
        pred = self.model(toks, repr_layers=self.repr_layers, return_contacts=False, colossal = True)
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

    def on_fit_start(self) -> None:
        if self.cuda_mem_fraction < 1.0:
            colo_set_process_memory_fraction(self.cuda_mem_fraction)

class tuner_ESM(pl.LightningModule):
    def __init__(self, model, lr, leggo = False):
        super().__init__()
        self.esm, self.alphabet = model
        self.repr_layers = [(i + self.esm.num_layers + 1) % (self.esm.num_layers + 1) for i in [-1]]
        self.reps = []
        self.lr = lr
        self.sample_seqs = []
        self.max_size = 0
        if leggo:
            self.leggo = True
        else:
            self.leggo = False
        self.optimizer = None

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        labels, seqs, toks = batch
        size = len(seqs[0])
        del labels, seqs, batch_idx
        masked_toks = maskInputs(toks)
        # try:
        output = self.esm(masked_toks, repr_layers = [-1], return_contacts=False)
        loss = F.cross_entropy(output['logits'].permute(0,2,1), toks)
        # print(size)
        self.max_size = size
        # except Exception as e:
        #     # self.max_size = len(masked_toks[0])
        #     raise Exception(e)
        del masked_toks, toks
        self.max_size = size
        return {"loss": loss}
    
    def configure_optimizers(self):
        if self.leggo:
            optimizer = DeepSpeedCPUAdam(self.esm.parameters(), lr=self.lr)
            return optimizer
        else:
            optimizer = torch.optim.Adam(self.esm.parameters(), lr=self.lr)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            self.optimizer = optimizer
            return [optimizer], [lr_scheduler]
    
    def predict_step(self, batch, batch_idx):
        labels, seqs, toks = batch
        try:
            pred = self.esm(toks, repr_layers=self.repr_layers, return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in pred["representations"].items()}
            rep_numpy = representations[self.repr_layers[0]].cpu().detach().numpy()
            reps = []
            for i in range(len(rep_numpy)):
                # self.reps.append(tuple([rep_numpy[i].mean(0), labels[i]]))
                reps.append(tuple([rep_numpy[i].mean(0), labels[i]]))
        except Exception as e:
            self.max_size = len(seqs[0])
            raise Exception(e)
        # newdf = pd.DataFrame(reps, columns = ['Embeddings', 'Label'])
        # finaldf = newdf['Embeddings'].apply(pd.Series)
        # finaldf['Label'] = newdf['Label']
        # return finaldf
        return reps
    #https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
    def wipe_memory(self): # DOES WORK
        self._optimizer_to(torch.device('cpu'))
        del self.optimizer
        gc.collect()
        torch.cuda.empty_cache()

    def _optimizer_to(self, device):
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    
    
class ProtGPT2(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
        self.model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2")
        self.lr = lr
        self.max_size = 0
        self.optimizer = None
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
        self.max_size = len(batch["Labels"][0])
        self.log("loss", loss)
        return(loss)
        
    def configure_optimizers(self):
        optimizer = DeepSpeedCPUAdam(self.model.parameters(), lr=1e-5)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.optimizer = optimizer
        # optimizer = FusedAdam(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def generate(self, seed_seq = "M", max_length = 100, do_sample = True, top_k = 950, repetition_penalty = 1.2, num_return_sequences = 5, eos_token_id=0):
        generator = pipeline('text-generation', model = self.model, tokenizer=self.tokenizer)
        outseqs = generator(seed_seq, max_length=max_length, do_sample =do_sample, top_k=top_k, repetition_penalty=repetition_penalty, num_return_sequences=num_return_sequences, eos_token_id=eos_token_id)
        outseqs = [samp['generated_text'].replace('\n','') for samp in outseqs]
        return outseqs

    def wipe_memory(self): # DOES WORK
        self._optimizer_to(torch.device('cpu'))
        del self.optimizer
        gc.collect()
        torch.cuda.empty_cache()

    def _optimizer_to(self, device):
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)


# class Protein_MPNN(pl.LightningModule):
#     def __init__(self, model, temp):
#         super().__init__()
#         subprocess.run(["wget", "-nc", f"https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/vanilla_model_weights/{model}.pt"])
#         checkpoint = torch.load(f'{model}.pt')
#         self.model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.0, k_neighbors=checkpoint['num_edges'])
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.temp = temp

#     def training_step(self, batch, batch_idx):
#         torch.cuda.empty_cache()
#         return {"loss": 0}
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
#         lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
#         return [optimizer], [lr_scheduler]
    
#     def predict_step(self, batch, batch_idx):
#         tied_positions_dict = None
#         chain_id_dict = None
#         fixed_positions_dict = None
#         pssm_dict = None
#         bias_by_res_dict = None
#         omit_AA_dict = None
#         batch_clones = [copy.deepcopy(batch) for i in range(1)]
#         X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, 'cuda', chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict)
#         randn_1 = torch.randn(chain_M.shape)
#         log_probs = self.model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
#         mask_for_loss = mask*chain_M*chain_M_pos
#         scores = _scores(S, log_probs, mask_for_loss)
#         native_score = scores.cpu().data.numpy()
#         global_scores = _scores(S, log_probs, mask)
#         global_native_score = global_scores.cpu().data.numpy()


#         for j in range(1):
#             randn_2 = torch.randn(chain_M.shape)
#             if tied_positions_dict == None:
#                 sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=self.temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0,pssm_log_odds_mask=pssm_log_odds_mask, bias_by_res=bias_by_res_all)
#                 S_sample = sample_dict["S"] 

#             log_probs = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
#             mask_for_loss = mask*chain_M*chain_M_pos
#             scores = _scores(S_sample, log_probs, mask_for_loss)
#             scores = scores.cpu().data.numpy()
            
#             global_scores = _scores(S_sample, log_probs, mask) #score the whole structure-sequence
#             global_scores = global_scores.cpu().data.numpy()
            
#             all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
#             all_log_probs_list.append(log_probs.cpu().data.numpy())
#             S_sample_list.append(S_sample.cpu().data.numpy())
#             for b_ix in range(1):
#                 masked_chain_length_list = masked_chain_length_list_list[b_ix]
#                 masked_list = masked_list_list[b_ix]
#                 seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21),axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
#                 seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
#                 score = scores[b_ix]
#                 score_list.append(score)
#                 global_score = global_scores[b_ix]
#                 global_score_list.append(global_score)
#                 native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
#                 if b_ix == 0 and j==0:
#                     start = 0
#                     end = 0
#                     list_of_AAs = []
#                     for mask_l in masked_chain_length_list:
#                         end += mask_l
#                         list_of_AAs.append(native_seq[start:end])
#                         start = end
#                     native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
#                     l0 = 0
#                     for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
#                         l0 += mc_length
#                         native_seq = native_seq[:l0] + '/' + native_seq[l0:]
#                         l0 += 1
#                     sorted_masked_chain_letters = np.argsort(masked_list_list[0])
#                     print_masked_chains = [masked_list_list[0][i] for i in sorted_masked_chain_letters]
#                     sorted_visible_chain_letters = np.argsort(visible_list_list[0])
#                     print_visible_chains = [visible_list_list[0][i] for i in sorted_visible_chain_letters]
#                     native_score_print = np.format_float_positional(np.float32(native_score.mean()), unique=False, precision=4)
#                     global_native_score_print = np.format_float_positional(np.float32(global_native_score.mean()), unique=False, precision=4)
#                     # script_dir = os.path.dirname(os.path.realpath(__file__))
#                     # try:
#                     #     commit_str = subprocess.check_output(f'git --git-dir {script_dir}/.git rev-parse HEAD', shell=True, stderr=subprocess.DEVNULL).decode().strip()
#                     # except subprocess.CalledProcessError:
#                     #     commit_str = 'unknown'
#                     # if args.ca_only:
#                     #     print_model_name = 'CA_model_name'
#                     # else:
#                     #     print_model_name = 'model_name'
#                     # f.write('>{}, score={}, global_score={}, fixed_chains={}, designed_chains={}, {}={}, git_hash={}, seed={}\n{}\n'.format(name_, native_score_print, global_native_score_print, print_visible_chains, print_masked_chains, print_model_name, args.model_name, commit_str, seed, native_seq)) #write the native sequence
#                 start = 0
#                 end = 0
#                 list_of_AAs = []
#                 for mask_l in masked_chain_length_list:
#                     end += mask_l
#                     list_of_AAs.append(seq[start:end])
#                     start = end

#                 seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
#                 l0 = 0
#                 for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
#                     l0 += mc_length
#                     seq = seq[:l0] + '/' + seq[l0:]
#                     l0 += 1
#                 score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
#                 global_score_print = np.format_float_positional(np.float32(global_score), unique=False, precision=4)
#                 seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4)
#                 sample_number = j*1+b_ix+1
#                     # f.write('>T={}, sample={}, score={}, global_score={}, seq_recovery={}\n{}\n'.format(temp,sample_number,score_print,global_score_print,seq_rec_print,seq)) #write generated sequence
#         # if args.save_score:
#         #     np.savez(score_file, score=np.array(score_list, np.float32), global_score=np.array(global_score_list, np.float32))
#         # if args.save_probs:
#         #     all_probs_concat = np.concatenate(all_probs_list)
#         #     all_log_probs_concat = np.concatenate(all_log_probs_list)
#         #     S_sample_concat = np.concatenate(S_sample_list)
#         #     np.savez(probs_file, probs=np.array(all_probs_concat, np.float32), log_probs=np.array(all_log_probs_concat, np.float32), S=np.array(S_sample_concat, np.int32), mask=mask_for_loss.cpu().data.numpy(), chain_order=chain_list_list)
#         print(seq)
#         return seq

    
    
from pytorch_lightning.callbacks import BasePredictionWriter

class CustomWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))