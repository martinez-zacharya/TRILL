import multiprocessing
import os
import subprocess

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import esm
import time
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import EsmForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
from evaluate import load
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer
from skopt import BayesSearchCV
from icecream import ic
from skopt.space import Real, Categorical, Integer
from trill.utils.logging import setup_logger
import requests
from Bio import SeqIO
from loguru import logger
from tqdm import tqdm

def load_data(args):
    if not args.preComputed_Embs:
        embed_command = f"trill {args.name} {args.GPUs} --outdir {args.outdir} embed {args.emb_model} {args.query} --avg"
        subprocess.run(embed_command.split(' '), check=True)
        df = pd.read_csv(os.path.join(args.outdir, f'{args.name}_{args.emb_model}_AVG.csv'))
    else:
        df = pd.read_csv(args.preComputed_Embs)
    return df

def load_model(args):
    # Check the model type and load accordingly
    if args.classifier == 'XGBoost':
        clf = xgb.XGBClassifier()
        clf.load_model(args.preTrained)  
    elif args.classifier == 'LightGBM':
        clf = lgb.Booster(model_file=args.preTrained)  
    else:
        logger.error("Unsupported model type. Please specify 'XGBoost' or 'LightGBM'.")
        raise ValueError("Unsupported model type. Please specify 'XGBoost' or 'LightGBM'.")

    return clf



def prep_data(df, args):
    if args.train_split is not None:
        if args.key == None:
            logger.error('Training a classifier requires a class key CSV!')
            raise Exception('Training a classifier requires a class key CSV!')
        key_df = pd.read_csv(args.key)
        n_classes = key_df['Class'].nunique()
        df['NewLab'] = None
        df['NewLab'] = df['NewLab'].astype(object)
        for cls in key_df['Class'].unique():
            condition = key_df[key_df['Class'] == cls]['Label'].tolist()
            df.loc[df['Label'].isin(condition), 'NewLab'] = cls
        df = df.sample(frac=1)
        train_df, test_df = train_test_split(df, train_size=float(args.train_split), stratify=df['NewLab'])
    return train_df, test_df, n_classes


def prep_hf_data(args):
    data = esm.data.FastaBatchedDataset.from_file(args.query)
    labels = data[:][0]
    sequences = data[:][1]
    seq_df = pd.DataFrame({'Label': labels, 'Sequence': sequences})
    train_df, test_df, n_classes = prep_data(seq_df, args)
    le = LabelEncoder()
    train_df["NewLab"] = le.fit_transform(train_df["NewLab"])
    test_df["NewLab"] = le.transform(test_df["NewLab"])

    return train_df, test_df, n_classes, le

def binary_compute_f1(eval_pred):
    f1_metric = load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1_metric.compute(predictions=predictions, references=labels, average='binary') 

def compute_f1(eval_pred):
    f1_metric = load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1_metric.compute(predictions=predictions, references=labels, average='macro') 

def setup_esm2_hf(train, test, args, n_classes):
    if float(args.lr) == 0.2:
        args.lr = 0.0001
    model = EsmForSequenceClassification.from_pretrained(f"facebook/{args.emb_model}_UR50D", use_safetensors=False, num_labels=n_classes)
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.emb_model}_UR50D")
    train_tokenized = tokenizer(train['Sequence'].to_list())
    test_tokenized = tokenizer(test['Sequence'].to_list())

    train_dataset = Dataset.from_dict(train_tokenized)
    test_dataset = Dataset.from_dict(test_tokenized)

    train_class_list = [[item] for item in train['NewLab']]
    test_class_list = [[item] for item in test['NewLab']]

    train_dataset = train_dataset.add_column("label", train_class_list)
    test_dataset = test_dataset.add_column("label", test_class_list)
    if int(args.GPUs) > 0:
        use_cpu = False
        fp16 = True
    else:
        use_cpu = True
        fp16 = False

    config = TrainingArguments(
        os.path.join(args.outdir, f"{args.name}_{args.emb_model}-MLP_{n_classes}-classifier.pt"),
        # evaluation_strategy = "epoch",
        save_strategy = "no",
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.batch_size),
        num_train_epochs=int(args.epochs),
        seed = int(args.RNG_seed),
        # load_best_model_at_end=True,
        # metric_for_best_model="f1",
        use_cpu = use_cpu, 
        # log_level='debug',
        fp16 = fp16
    )
    if n_classes == 2:
        trainer = Trainer(
        model,
        config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=binary_compute_f1,
        )
    else:
        trainer = Trainer(
        model,
        config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_f1,
        ) 
    logger.info("Starting training of ESM2+MLP!")
    return trainer, test_dataset

def custom_esm2mlp_test(args):
    model = EsmForSequenceClassification.from_pretrained(args.preTrained, use_safetensors=False)
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.emb_model}_UR50D")
    n_classes = model.config.num_labels
    data = esm.data.FastaBatchedDataset.from_file(args.query)
    labels = data[:][0]
    sequences = data[:][1]
    label_list = [[item] for item in labels]
    seq_df = pd.DataFrame({'Label': labels, 'Sequence': sequences})
    train_tokenized = tokenizer(seq_df['Sequence'].to_list())
    dataset = Dataset.from_dict(train_tokenized)
    # dataset = dataset.add_column("labels", seq_df['Label'].to_list())
    if int(args.GPUs) > 0:
        use_cpu = False
        fp16 = True
    else:
        use_cpu = True
        fp16 = False

    config = TrainingArguments(
        os.path.join(args.outdir, f"{args.name}_{args.emb_model}-MLP_{n_classes}-classifier"),
        # evaluation_strategy = "epoch",
        save_strategy = "no",
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.batch_size),
        num_train_epochs=int(args.epochs),
        seed = int(args.RNG_seed),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        use_cpu = use_cpu, 
        # log_level='debug',
        fp16 = fp16
    )
    trainer = Trainer(
        model,
        config,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        ) 
    return trainer, dataset, seq_df['Label'].to_list()

def train_model(train_df, args):
    if args.classifier == 'LightGBM':
        d_train = lgb.Dataset(train_df.iloc[:, :-2], label=train_df['NewLab'])
    elif args.classifier == 'XGBoost':
        d_train = xgb.DMatrix(train_df.iloc[:, :-2], label=train_df['NewLab'])
    
    config = {
        'lightgbm': {
            'objective': 'multiclass',
            'num_class': len(train_df['NewLab'].unique()),
            'metric': 'multi_logloss',
            'learning_rate': args.lr,
            'max_depth': args.max_depth,
            'num_leaves': args.num_leaves,
            'feature_fraction': args.feature_frac,
            'bagging_fraction': args.bagging_frac,
            'bagging_freq': args.bagging_freq,
            'num_threads': args.n_workers,
            'seed': args.RNG_seed,
            'verbosity' : -1
        },
        'xgboost': {
            'objective': 'multi:softprob',
            'num_class': len(train_df['NewLab'].unique()),
            'learning_rate': args.lr,
            'max_depth': args.max_depth,
            'n_estimators': args.n_estimators,
            'gamma': args.xg_gamma,
            'reg_alpha': args.xg_reg_alpha,
            'reg_lambda': args.xg_reg_lambda,
            'seed': args.RNG_seed,
            'nthread': args.n_workers
        }
    }
    
    # Model training
    if args.classifier == 'LightGBM':
        clf = lgb.train(config['lightgbm'], d_train, args.n_estimators)
    elif args.classifier == 'XGBoost':
        clf = xgb.train(config['xgboost'], d_train, args.n_estimators)
    
    return clf


def predict_and_evaluate(model, le, test_df, args):
    if args.classifier == 'LightGBM':
        test_preds = model.predict(test_df.iloc[:, :-2])
        # test_preds = np.argmax(test_preds, axis=0)
    elif args.classifier == 'XGBoost':
        # d_test = xgb.DMatrix(test_df.iloc[:, :-2])
        # test_preds = model.predict(d_test)
        test_preds = model.predict(test_df.iloc[:, :-2])
        # test_preds = np.argmax(test_preds, axis=1)
    transformed_preds = le.inverse_transform(test_preds)
    precision, recall, fscore, support = precision_recall_fscore_support(test_df['NewLab'].values, test_preds, average=args.f1_avg_method,labels=np.unique(test_df['NewLab']))
                                                                         
    return precision, recall, fscore, support

def custom_model_test(model, test_df, args):
    # Generate probability predictions based on the model type
    model_type = args.classifier
    if model_type == 'XGBoost':
        test_preds_proba = model.predict_proba(test_df.iloc[:, :-1])
        proba_df = pd.DataFrame(test_preds_proba)
        test_preds = proba_df.idxmax(axis=1)
    elif model_type == 'LightGBM':
        test_preds_proba = model.predict(test_df.iloc[:, :-1], raw_score=True)
        proba_df = pd.DataFrame(test_preds_proba)
        if test_preds_proba.ndim == 1:
            test_preds = (test_preds_proba > 0).astype(int)
        else:
            test_preds = proba_df.idxmax(axis=1)

    # Add the original labels to the DataFrame
    proba_df['Label'] = test_df['Label'].values
    
    # Save the probabilities to a CSV file
    proba_file_name = f'{args.name}_{model_type}_class_probs.csv'
    proba_df.to_csv(os.path.join(args.outdir, proba_file_name), index=False)
    
    # Prepare and save the predictions to a CSV file
    pred_df = pd.DataFrame(test_preds, columns=['Prediction'])
    pred_df['Label'] = test_df['Label']
    
    pred_file_name = f'{args.name}_{model_type}_predictions.csv'
    pred_df.to_csv(os.path.join(args.outdir, pred_file_name), index=False)

    return

def custom_xg_test(model, test_df, args):
    test_preds_proba = model.predict_proba(test_df.iloc[:, :-1])
    proba_df = pd.DataFrame(test_preds_proba)
    test_preds = proba_df.idxmax(axis=1)
    proba_df['Label'] = test_df['Label'].values
    proba_df.to_csv(os.path.join(args.outdir, f'{args.name}_XGBoost_class_probs.csv'), index=False)
    pred_df = pd.DataFrame(test_preds)
    pred_df['Label'] = test_df['Label']
    pred_df.to_csv(os.path.join(args.outdir, f'{args.name}_XGBoost_predictions.csv'), index=False)
    return 

def log_results(out_file, command_str, n_classes, args, classes = None, sweeped_clf=None, precision=None, recall=None, fscore=None, support=None, le=None):
    with open(out_file, 'w+') as out:
        out.write('TRILL command used: ' + command_str + '\n\n')
        out.write(f'Classes trained on: {classes}\n\n')

        if sweeped_clf and args.f1_avg_method != None:
            out.write(f'Best sweep params: {sweeped_clf.best_params_}\n\n')
            out.write(f'Best sweep F1 score: {sweeped_clf.best_score_}\n\n')
            out.write(f"{args.f1_avg_method}-averaged classification metrics:\n")
            out.write(f"\tPrecision: {precision}\n")
            out.write(f"\tRecall: {recall}\n")
            out.write(f"\tF-score: {fscore}\n")
        elif sweeped_clf and args.f1_avg_method == None:
            out.write(f'Best sweep params: {sweeped_clf.best_params_}\n\n')
            out.write(f'Best sweep F1 score: {sweeped_clf.best_score_}\n\n')
            out.write("Classification Metrics Per Class:\n")
            for i, label in enumerate(classes):
                num_label = le.transform([label])
                out.write(f"\nClass {num_label} : {label}\n")
                
                out.write(f"\tPrecision: {precision[i]}\n")
                
                out.write(f"\tRecall: {recall[i]}\n")
                
                out.write(f"\tF-score: {fscore[i]}\n")

                out.write(f"\tSupport: {support[i]}\n")
        elif precision is not None and recall is not None and fscore is not None and support is not None:  
            out.write("Classification Metrics Per Class:\n")
            for i, label in enumerate(classes):
                num_label = le.transform([label])
                out.write(f"\nClass {num_label} : {label}\n")
                
                out.write(f"\tPrecision: {precision[i]}\n")
                
                out.write(f"\tRecall: {recall[i]}\n")
                
                out.write(f"\tF-score: {fscore[i]}\n")

                out.write(f"\tSupport: {support[i]}\n")

            # Compute and display average metrics
            avg_precision = np.mean(precision)
            avg_recall = np.mean(recall)
            avg_fscore = np.mean(fscore)

            out.write("\nAverage Metrics:\n")

            out.write(f"\tAverage Precision: {avg_precision:.4f}\n")

            out.write(f"\tAverage Recall: {avg_recall:.4f}\n")

            out.write(f"\tAverage F-score: {avg_fscore:.4f}\n")
        elif precision is not None and recall is not None and fscore is not None:
            out.write("Classification Metrics Per Class:\n")
            for i in n_classes:
                out.write(f"\nClass: {label}\n")
                out.write(f"\tPrecision: {precision[i]}\n")
                out.write(f"\tRecall: {recall[i]}\n")
                out.write(f"\tF-score: {fscore[i]}\n")

def generate_class_key_csv(args):
    all_headers = []
    all_labels = []
    
    # If directory is provided
    if args.dir:
        for filename in os.listdir(args.dir):
            if filename.endswith('.fasta'):
                class_label = os.path.splitext(filename)[0]
                
                with open(os.path.join(args.dir, filename), 'r') as fasta_file:
                    for line in fasta_file:
                        line = line.strip()
                        if line.startswith('>'):
                            all_headers.append(line[1:])
                            all_labels.append(class_label)
    
    # If text file with paths is provided
    elif args.fasta_paths_txt:
        with open(args.fasta_paths_txt, 'r') as txt_file:
            for path in txt_file:
                path = path.strip()
                if not path:  # Skip empty or whitespace-only lines
                    continue
                
                class_label = os.path.splitext(os.path.basename(path))[0]
                
                if not os.path.exists(path):
                    logger.warning(f"File {path} does not exist.")
                    continue
                
                with open(path, 'r') as fasta_file:
                    for line in fasta_file:
                        line = line.strip()
                        if line.startswith('>'):
                            all_headers.append(line[1:])
                            all_labels.append(class_label)
    else:
        logger.warning('prepare_class_key requires either a path to a directory of fastas or a text file of fasta paths!')
        raise RuntimeError
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'Label': all_headers,
        'Class': all_labels
    })
    outpath = os.path.join(args.outdir, f'{args.name}_class_key.csv')
    df.to_csv(outpath, index=False)
    logger.info(f"Class key CSV generated and saved as '{outpath}'.")



def sweep(train_df, args):
    model_type = args.classifier
    logger.info(f"Setting up hyperparameter sweep for {model_type}")
    np.int = np.int64
    if args.n_workers == 1:
        logger.warning("WARNING!")
        logger.warning("You are trying to perform a hyperparameter sweep with only 1 core!")
        logger.warning(f"In your case, you have {multiprocessing.cpu_count()} CPU cores available!")
    logger.info(f"Using {args.n_workers} CPU cores for sweep")
    # Define model and parameter grid based on the specified model_type
    if model_type == 'LightGBM':
        if train_df['NewLab'].nunique() == 2:
            objective = 'binary'
            f1_avg_method = 'binary'
        else:
            objective = 'multiclass'
            f1_avg_method = 'macro'
        model = lgb.LGBMClassifier(objective=objective, verbose=-1)
        param_grid = {
            'learning_rate': Real(0.01, 0.3),
            'num_leaves': Integer(10, 255),
            'max_depth': Integer(-1, 20),
            'n_estimators': Integer(50, 250),
            'feature_fraction': Real(0.1, 1.0),
            'bagging_fraction': Real(0, 1),
            'bagging_freq': Integer(0, 10),
        }
    elif model_type == 'XGBoost':
        if train_df['NewLab'].nunique() == 2:
            model = xgb.XGBClassifier(objective='binary:logitraw')
            f1_avg_method = 'binary'
        else:
            model = xgb.XGBClassifier(objective='multi:softprob')
            f1_avg_method = 'macro'
        param_grid = {
            'booster': Categorical(['gbtree']),
            'gamma': Real(0, 5),
            'learning_rate': Real(0.01, 0.3),
            'max_depth': Integer(5, 20),
            'n_estimators': Integer(50, 250),
            'reg_alpha': Real(0, 1),
            'reg_lambda': Real(0, 1),
            'subsample': Real(0.5, 1.0),
            'colsample_bytree': Real(0.5, 1.0),
            'min_child_weight': Integer(1, 10),
        }
    
    f1_scorer = make_scorer(f1_score, average=f1_avg_method)
    
    clf = BayesSearchCV(estimator=model, search_spaces=param_grid, n_iter=10, n_jobs=int(args.n_workers),scoring=f1_scorer, cv=int(args.sweep_cv), return_train_score=True, verbose=1)
    
    logger.info("Sweeping...")
    clf.fit(train_df.iloc[:, :-2], train_df['NewLab'])
    
    # Save the best model
    if model_type == 'LightGBM':
        clf.best_estimator_.booster_.save_model(os.path.join(args.outdir, f'{args.name}_LightGBM_sweeped.json'))
    elif model_type == 'XGBoost':
        clf.best_estimator_.save_model(os.path.join(args.outdir, f'{args.name}_XGBoost_sweeped.json'))
    
    logger.info("Sweep Complete! Now evaluating...")
    
    return clf


# From mheinzinger at https://github.com/mheinzinger/ProstT5/blob/main/scripts/predict_3Di_encoderOnly.py
# Convolutional neural network (two convolutional layers)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        """
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of classes (20 for 3Di)
        """
        x = x.permute(0, 2, 1).unsqueeze(
            dim=-1)  # IN: X = (B x L x F); OUT: (B x F x L, 1)
        Yhat = self.classifier(x)  # OUT: Yhat_consurf = (B x N x L x 1)
        Yhat = Yhat.squeeze(dim=-1)  # IN: (B x N x L x 1); OUT: ( B x N x L )
        return Yhat


def get_T5_model():
    model = T5EncoderModel.from_pretrained(
        "Rostlab/ProstT5_fp16")
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(
        "Rostlab/ProstT5_fp16", do_lower_case=False)
    return model, vocab


def read_fasta(fasta_path):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''

    sequences = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line[1:]
                sequences[uniprot_id] = ''
            else:
                s = ''.join(line.split()).replace("-", "")

                if s.islower():  # sanity check to avoid mix-up of 3Di and AA input
                    print("The input file was in lower-case which indicates 3Di-input." +
                          "This predictor only operates on amino-acid-input (upper-case)." +
                          "Exiting now ..."
                          )
                    return None
                else:
                    sequences[uniprot_id] += s
    return sequences


def write_probs(predictions, args, input):
    out_path = os.path.join(args.outdir, f"{args.name}_{input}_3Di_output_probabilities.csv")
    
    # Write to the output file
    with open(out_path, 'w+') as out_f:
        for seq_id, (_, prob) in predictions.items():
            # Write each line as a formatted string, ensuring CSV compatibility
            out_f.write(f'{seq_id.strip()},{prob}\n')
    
    # Assuming logger is configured elsewhere in your application
    logger.info(f"Finished writing probabilities to {out_path}")
    return out_path


def write_predictions(predictions, args, input):
    ss_mapping = {
        0: "A",
        1: "C",
        2: "D",
        3: "E",
        4: "F",
        5: "G",
        6: "H",
        7: "I",
        8: "K",
        9: "L",
        10: "M",
        11: "N",
        12: "P",
        13: "Q",
        14: "R",
        15: "S",
        16: "T",
        17: "V",
        18: "W",
        19: "Y"
    }
    out_path = os.path.join(args.outdir, f'{args.name}_{input}_3Di.fasta')
    with open(out_path, 'w+') as out_f:
        out_f.write('\n'.join(
            [">{}{}".format(
                seq_id, "".join(list(map(lambda yhat: ss_mapping[int(yhat)], yhats))))
             for seq_id, (yhats, _) in predictions.items()
             ]
        ))
    logger.info(f"Finished writing results to {out_path}")
    return out_path


def toCPU(tensor):
    if len(tensor.shape) > 1:
        return tensor.detach().cpu().squeeze(dim=-1).numpy()
    else:
        return tensor.detach().cpu().numpy()



def load_predictor(args, cache_dir, weights_link="https://github.com/mheinzinger/ProstT5/raw/main/cnn_chkpnt/model.pt"):
    model = CNN()
    checkpoint_p = os.path.join(cache_dir, 'ProstT5_3Di_CNN.pt')
    # if no pre-trained model is available, yet --> download it
    if not os.path.exists(weights_link):
        r = requests.get(weights_link)
        logger.info('Downloading ProstT5 3Di CNN weights...')
        with open(checkpoint_p, "wb") as file:
            file.write(r.content)
        logger.info('Finished downloading ProstT5 3Di CNN weights!')


    state = torch.load(checkpoint_p)

    model.load_state_dict(state["state_dict"])

    model = model.eval()
    # model = model.to(device)

    return model


def get_3di_embeddings(args, cache_dir,
                   max_residues=4000, max_seq_len=4000, max_batch=500):

    seq_dict = dict()
    predictions = dict()

    # Read in fasta
    prefix = "<AA2fold>"

    model, vocab = get_T5_model()
    predictor = load_predictor(args, cache_dir)

    # if half_precision:
    if int(args.GPUs) >= 1:
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)
    predictor.to(device)
    if int(args.GPUs) >= 1:
        logger.info("Using models in half-precision.")
        model.half()
        predictor.half()
    # else:
    #     model.to(torch.float32)
    #     predictor.to(torch.float32)
    #     print("Using models in full-precision.")
    pred_path_list = []
    for input in ['query', 'db']:
        current_input = getattr(args, input)
        seq_dict = read_fasta(current_input)
        logger.info('Total number of sequences: {}'.format(len(seq_dict)))


        # sort sequences by length to trigger OOM at the beginning
        seq_dict = sorted(seq_dict.items(), key=lambda kv: len(
            seq_dict[kv[0]]), reverse=True)

        batch = list()
        standard_aa = "ACDEFGHIKLMNPQRSTVWY"
        standard_aa_dict = {aa: aa for aa in standard_aa}
        for seq_idx, (pdb_id, seq) in tqdm(enumerate(seq_dict, 1)):
            # replace the non-standard amino acids with 'X'
            seq = ''.join([standard_aa_dict.get(aa, 'X') for aa in seq])
            #seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
            seq_len = len(seq)
            seq = prefix + ' ' + ' '.join(list(seq))
            batch.append((pdb_id, seq, seq_len))

            # count residues in current batch and add the last sequence length to
            # avoid that batches with (n_res_batch > max_residues) get processed
            n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
            if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
                pdb_ids, seqs, seq_lens = zip(*batch)
                batch = list()

                token_encoding = vocab.batch_encode_plus(seqs,
                                                        add_special_tokens=True,
                                                        padding="longest",
                                                        return_tensors='pt'
                                                        ).to(device)
                try:
                    with torch.no_grad():
                        embedding_repr = model(token_encoding.input_ids,
                                            attention_mask=token_encoding.attention_mask
                                            )
                except RuntimeError:
                    print("RuntimeError during embedding for {} (L={})".format(
                        pdb_id, seq_len)
                    )
                    continue

                # ProtT5 appends a special tokens at the end of each sequence
                # Mask this also out during inference while taking into account the prefix
                for idx, s_len in enumerate(seq_lens):
                    token_encoding.attention_mask[idx, s_len+1] = 0

                # extract last hidden states (=embeddings)
                residue_embedding = embedding_repr.last_hidden_state.detach()
                # mask out padded elements in the attention output (can be non-zero) for further processing/prediction
                residue_embedding = residue_embedding * \
                    token_encoding.attention_mask.unsqueeze(dim=-1)
                # slice off embedding of special token prepended before to each sequence
                residue_embedding = residue_embedding[:, 1:]

                # IN: X = (B x L x F) - OUT: ( B x N x L )
                prediction = predictor(residue_embedding)
                probabilities = toCPU(torch.max(
                    F.softmax(prediction, dim=1), dim=1, keepdim=True)[0])
                
                prediction = toCPU(torch.max(prediction, dim=1, keepdim=True)[
                                1]).astype(np.byte)

                # batch-size x seq_len x embedding_dim
                # extra token is added at the end of the seq
                for batch_idx, identifier in enumerate(pdb_ids):
                    s_len = seq_lens[batch_idx]
                    # slice off padding and special token appended to the end of the sequence
                    pred = prediction[batch_idx, :, 0:s_len].squeeze()
                    prob = int( 100* np.mean(probabilities[batch_idx, :, 0:s_len]))
                    predictions[identifier] = (pred, prob)
                    assert s_len == len(predictions[identifier][0]), logger.warning(
                        f"Length mismatch for {identifier}: is:{len(predictions[identifier])} vs should:{s_len}")


        preds_out_path = write_predictions(predictions, args, input)
        probs_out_path = write_probs(predictions, args, input)
        pred_path_list.append(preds_out_path)

    return pred_path_list

def prep_foldseek_dbs(fasta_aa, fasta_3di, output_basename):
    sequences_aa = {}
    for record in SeqIO.parse(fasta_aa, "fasta"):
        sequences_aa[record.id] = str(record.seq)

    # read in 3Di strings
    sequences_3di = {}
    for record in SeqIO.parse(fasta_3di, "fasta"):
        sequences_3di[record.id] = str(record.seq).upper()

    # generate TSV file contents
    tsv_aa = ""
    tsv_3di = ""
    tsv_header = ""
    for i, id in enumerate(sequences_aa.keys()):
        tsv_aa += "{}\t{}\n".format(str(i + 1), sequences_aa[id])
        tsv_3di += "{}\t{}\n".format(str(i + 1), sequences_3di[id])
        tsv_header += "{}\t{}\n".format(str(i + 1), id)

    with open(f'tmp_{output_basename}_aa.tsv', "w+") as f:
        f.write(tsv_aa)
    with open(f'tmp_{output_basename}_tdi.tsv', "w+") as f:
        f.write(tsv_3di)
    with open(f'tmp_{output_basename}_header.tsv', "w+") as f:
        f.write(tsv_header)

    for type, code, suff in [('aa', 0, ''), ('tdi', 0, '_ss'), ('header', 12, '_h')]:
        foldseek_tsv2db = f'foldseek tsv2db tmp_{output_basename}_{type}.tsv tmp_{output_basename}_db{suff} --output-dbtype {code} -v 0'.split()
        subprocess.run(foldseek_tsv2db)
