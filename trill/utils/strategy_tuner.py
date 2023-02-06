import pytorch_lightning as pl
import torch
import esm
import os
import time
import pandas as pd
from trill.utils.lightning_models import tuner_ESM, CustomWriter, ProtGPT2
from trill.utils.protgpt2_utils import ProtGPT2_wrangle
from transformers import AutoTokenizer, EsmForProteinFolding
from tqdm import tqdm
# TO-DO: Create script to automatically detect which
# strategy and hyperparameters to use given the
# input, model selection and hardware

def tune_esm_inference(data, gpu, billions):

    limits = []

    if billions == True:
        ESM2_list = [
            'esm2_t48_15B_UR50D',
            'esm2_t36_3B_UR50D',
            'esm2_t33_650M_UR50D',
            'esm2_t30_150M_UR50D',
            'esm2_t12_35M_UR50D',
            'esm2_t6_8M_UR50D'
        ]
    else:
        ESM2_list = [
            'esm2_t33_650M_UR50D',
            'esm2_t30_150M_UR50D',
            'esm2_t12_35M_UR50D',
            'esm2_t6_8M_UR50D'
        ]


    ESM2_list.reverse()
    for esm2 in ESM2_list:
        torch.cuda.empty_cache()
        try:
            model_import_name = f'esm.pretrained.{esm2}()'
            model = tuner_ESM(eval(model_import_name), float(0.0001))
            dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = 1, num_workers=0, collate_fn=model.alphabet.get_batch_converter())
            pred_writer = CustomWriter(output_dir=".", write_interval="epoch")
            trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], devices=gpu, accelerator='gpu', num_nodes=1)
            len_pls = trainer.predict(model, dataloader)
            cwd_files = os.listdir()
            pt_files = [file for file in cwd_files if 'predictions_' in file]
            pred_embeddings = []
            for pt in pt_files:
                preds = torch.load(pt)
                for pred in preds:
                    for sublist in pred:
                        pred_embeddings.append(tuple([sublist[0][0], sublist[0][1]]))
                embedding_df = pd.DataFrame(pred_embeddings, columns = ['Embeddings', 'Label'])
                finaldf = embedding_df['Embeddings'].apply(pd.Series)
                finaldf['Label'] = embedding_df['Label']
                finaldf.to_csv(f'{esm2}.csv', index = False)
            for file in pt_files:
                os.remove(file)
            # print(f'{esm2} is able to be used on your current GPU!!!')
            os.remove(f'{esm2}.csv')

        except Exception as e:
            limits.append((esm2, model.max_size))
        print((esm2, model.max_size))
    return limits




def tune_esm_train(data, gpu):
    limits = []

    if billions == True:
        ESM2_list = [
            'esm2_t48_15B_UR50D',
            'esm2_t36_3B_UR50D',
            'esm2_t33_650M_UR50D',
            'esm2_t30_150M_UR50D',
            'esm2_t12_35M_UR50D',
            'esm2_t6_8M_UR50D'
        ]
    else:
        ESM2_list = [
            'esm2_t33_650M_UR50D',
            'esm2_t30_150M_UR50D',
            'esm2_t12_35M_UR50D',
            'esm2_t6_8M_UR50D'
        ]

    strat_list = [
        None,
        'deepspeed_stage_1',
        'deepspeed_stage_2',
        'deepspeed_stage_2_offload',
        'deepspeed_stage_3',
        'deepspeed_stage_3_offload'
    ]

    ESM2_list.reverse()
    for esm2 in ESM2_list:
        torch.cuda.empty_cache()
        for strat in strat_list:
            try:
                dataset = esm.data.FastaBatchedDataset.from_file(data)
                torch.cuda.empty_cache()
                model_import_name = f'esm.pretrained.{esm2}()'
                model = tuner_ESM(eval(model_import_name), float(0.0001))
                dataloader = torch.utils.data.DataLoader(dataset, shuffle = False, batch_size = 1, num_workers=0, collate_fn=model.alphabet.get_batch_converter())
                trainer = pl.Trainer(devices=gpu, accelerator='gpu', strategy = strat, max_epochs=1, num_nodes=1, precision = 16, enable_checkpointing=False, replace_sampler_ddp=False)        
                # time.sleep(30)
                trainer.fit(model=model, train_dataloaders=dataloader)
            except Exception as e:
                # print(e)
                limits.append((esm2, strat, model.max_size))
                model.wipe_memory()
            else:
                model.wipe_memory()
                del model, dataloader, dataset
            finally:
                model.wipe_memory()
                del model, dataloader, dataset
    return(limits)

def tune_protgpt2_train(data, gpu):
    limits = []
    strat_list = [
        # None,
        # 'deepspeed_stage_1',
        # 'deepspeed_stage_2',
        # 'deepspeed_stage_2_offload',
        # 'deepspeed_stage_3',
        'deepspeed_stage_3_offload'
    ]
    for strat in strat_list:
        torch.cuda.empty_cache()
        try:
            tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
            model = ProtGPT2(0.0001, tokenizer)
            seq_dict_df = ProtGPT2_wrangle(data, tokenizer)
            dataloader = torch.utils.data.DataLoader(seq_dict_df, shuffle = False, batch_size = 1, num_workers=0)
            trainer = pl.Trainer(devices=gpu, accelerator='gpu', max_epochs=1, num_nodes = 1,replace_sampler_ddp=False, precision = 16, strategy = strat)
            trainer.fit(model=model, train_dataloaders = dataloader)

        except Exception as e:
            print(e)
            limits.append(('ProtGPT2', strat, model.max_size))
            model.wipe_memory()
    return(limits)

def tune_esmfold(data, gpu):
    limit = 0

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", device_map="auto")
    model.esm = model.esm.half()
    # model.trunk.set_chunk_size(32)
    fold_df = pd.DataFrame(list(data), columns = ["Entry", "Sequence"])
    outputs = []
    with torch.no_grad():
        for input_ids in tqdm(fold_df.Sequence.tolist()):
            tokenized_input = tokenizer([input_ids], return_tensors="pt", add_special_tokens=False)['input_ids']
            tokenized_input = tokenized_input.clone().detach()
            prot_len = len(input_ids)
            # try:
            output = model(tokenized_input)
            limit = prot_len
            # except Exception as e:
            #     print(e)
    return limit
