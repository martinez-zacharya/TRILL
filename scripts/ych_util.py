import torch
import numpy as np
import math

def prepare_mlm_mask(alphabet,batch_tokens,mask_frac = 0.15 ,random_subfrac = 0.1, keep_subfrac = 0.1):
    '''
    token_seq: a tokenized sequence of an original protein sequence (for now do batch_size = 1)
    the function generate index of 15% of the sequence to be mask_token_index, then out of the 15%, 10% kept as original, 10% changed to a randomized. 
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_labels: ['U5C9Z2_9BACT/7-195']

batch_strs: ['HQDNVHARSLMGLVRNVFEQAGLEKTALDAVAVSSGPGSYTGLRIGVSVAKGLAYALDKPVIGVGTLEALAFRAIPFSDSTDTIIPMLDARRMEVYALVMDGLGDTLISPQPFILEDNPFMEYLEKGKVFFLGDGVPKSKEILSHPNSRFVPLFNSSQSIGELAYKKFLKADFESLAYFEPNYIKEFRI']
batch_tokens: tensor([[32, 21, 16, 13, 17,  7, 21,  5, 10,  8,  4, 20,  6,  4,  7, 10, 17,  7,
         18,  9, 16,  5,  6,  4,  9, 15, 11,  5,  4, 13,  5,  7,  5,  7,  8,  8,
          6, 14,  6,  8, 19, 11,  6,  4, 10, 12,  6,  7,  8,  7,  5, 15,  6,  4,
          5, 19,  5,  4, 13, 15, 14,  7, 12,  6,  7,  6, 11,  4,  9,  5,  4,  5,
         18, 10,  5, 12, 14, 18,  8, 13,  8, 11, 13, 11, 12, 12, 14, 20,  4, 13,
          5, 10, 10, 20,  9,  7, 19,  5,  4,  7, 20, 13,  6,  4,  6, 13, 11,  4,
         12,  8, 14, 16, 14, 18, 12,  4,  9, 13, 17, 14, 18, 20,  9, 19,  4,  9,
         15,  6, 15,  7, 18, 18,  4,  6, 13,  6,  7, 14, 15,  8, 15,  9, 12,  4,
          8, 21, 14, 17,  8, 10, 18,  7, 14,  4, 18, 17,  8,  8, 16,  8, 12,  6,
          9,  4,  5, 19, 15, 15, 18,  4, 15,  5, 13, 18,  9,  8,  4,  5, 19, 18,
          9, 14, 17, 19, 12, 15,  9, 18, 10, 12]])
    '''
    # exclued the cls first token
    seq_len  = batch_tokens.size(1) - 1
    tot_masked_num = int(seq_len * mask_frac)
    random_num = int(random_subfrac * tot_masked_num)
    keep_num = int(keep_subfrac * tot_masked_num)

    masked_num = tot_masked_num - random_num - keep_num
    # generate #masked_num indices from range 1 to batch_tokens.size(1) to be masked
    shuffled_seqind = torch.randperm(seq_len)+1

    target_ind =  shuffled_seqind[:tot_masked_num]
    true_aa = batch_tokens[:,target_ind]
    keep_id = shuffled_seqind[0:random_num]
    random_id = shuffled_seqind[random_num:random_num+random_num]
    mask_id = shuffled_seqind[random_num+random_num:tot_masked_num]
    batch_tokens[:,mask_id] = alphabet.mask_idx
    # randomize and substitute the AA at the indices of random_id
    batch_tokens[:,random_id] = torch.randint(1,26,(1,random_num))
    return true_aa,target_ind,batch_tokens
    
    
    
    
    
    
    
    