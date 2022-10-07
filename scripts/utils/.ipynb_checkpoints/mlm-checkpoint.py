import torch
import numpy as np
import math

def prepare_mlm_mask(alphabet,batch_tokens,mask_frac = 0.15 ,random_subfrac = 0.1, keep_subfrac = 0.1):
    
    true_aa = batch_tokens.detach().clone()
#     true_aa = batch_tokens.detach()

    rand = torch.rand(batch_tokens.shape)
    mask_arr = (rand < 0.15) * (batch_tokens != 1) * (batch_tokens != 32)
    selection = []

    for i in range(batch_tokens.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
    )
    for i in range(batch_tokens.shape[0]):
        batch_tokens[i, selection[i]] = alphabet.mask_idx
    del rand, mask_arr, selection
    return(true_aa, batch_tokens)