import pytorch_lightning as pl
import torch
import esm
import os
import pandas as pd
from trill.utils.lightning_models import ESM, CustomWriter
# TO-DO: Create script to automatically detect which
# strategy and hyperparameters to use given the
# input, model selection and hardware

def tune_esm_inference(data):
    ESM2_list = [
        'esm2_t48_15B_UR50D',
        'esm2_t36_3B_UR50D',
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
            model = ESM(eval(model_import_name), float(0.0001))
            dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = 1, num_workers=0, collate_fn=model.alphabet.get_batch_converter())
            pred_writer = CustomWriter(output_dir=".", write_interval="epoch")
            trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], devices=1, accelerator='gpu', num_nodes=1)
            trainer.predict(model, dataloader)
            
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
            print(f'{esm2} is able to be used on your current GPU!!!')
            os.remove(f'{esm2}.csv')

        except Exception as e:
            print(e)
            print(f'{esm2} is too big for current GPU...')
            break

def tune_esm_train(data, gpu):
    ESM2_list = [
        'esm2_t48_15B_UR50D',
        'esm2_t36_3B_UR50D',
        'esm2_t33_650M_UR50D',
        'esm2_t30_150M_UR50D',
        'esm2_t12_35M_UR50D',
        'esm2_t6_8M_UR50D'
    ]

    strat_list = [
        None,
        'DDPFullyShardedNativeStrategy',
        'deepspeed_stage_1',
        'deepspeed_stage_2',
        'deepspeed_stage_2',
        'deepspeed_stage_3',
        'deepspeed_stage_3_offload'
    ]

    ESM2_list.reverse()
    for esm2 in ESM2_list:
        is_trainable = False
        torch.cuda.empty_cache()
        for strat in strat_list:
            torch.cuda.empty_cache()
            if is_trainable == True:
                print('trainable')
                break
            try:
                model_import_name = f'esm.pretrained.{esm2}()'
                model = ESM(eval(model_import_name), float(0.0001))
                dataloader = torch.utils.data.DataLoader(data, shuffle = False, batch_size = 1, num_workers=0, collate_fn=model.alphabet.get_batch_converter())
                trainer = pl.Trainer(devices=gpu, accelerator='gpu', strategy = strat, max_epochs=1, num_nodes=1, precision = 16, amp_backend='native', enable_checkpointing=False)        
                trainer.fit(model=model, train_dataloaders=dataloader)
                is_trainable = True            
            except Exception as e:
                print(e)
                print(f'Unable to finetune {esm2} using {strat}...')

        if is_trainable == False:
            print(f'{esm2} and bigger models are not able to run using the current setup')