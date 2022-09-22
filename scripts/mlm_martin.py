import torch
import numpy as np

def maskInputs(train_inputs):

    # create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(train_inputs['input_ids'].shape)
    # create mask array
    mask_arr = (rand < 0.15) * (train_inputs['input_ids'] != 0) * \
              (train_inputs['input_ids'] != 2) * (train_inputs['input_ids'] != 1)

    selection = []

    for i in range(train_inputs['input_ids'].shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )
    selection[:5]

    for i in range(train_inputs['input_ids'].shape[0]):
        train_inputs['input_ids'][i, selection[i]] = 32

    train_inputs['input_ids']
    
    return train_inputs