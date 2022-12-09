import pandas as pd
import esm
import torch
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, AutoTokenizer


def ProtGPT2_wrangle(data, tokenizer):
    seqs_for_dl = []
    for pair in data:
        seqs_for_dl.append(tuple(pair))
    seq_dict = dict(seqs_for_dl)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm=False)
    seq_dict_df = pd.DataFrame(seq_dict.items(), columns = ['input_ids', 'Labels'])
    seq_dict_df = Dataset.from_pandas(seq_dict_df)
    return seq_dict_df