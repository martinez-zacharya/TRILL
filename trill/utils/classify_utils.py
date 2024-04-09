import multiprocessing
import os
import subprocess

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import esm
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import EsmForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
from evaluate import load
from skopt import BayesSearchCV
from icecream import ic
from skopt.space import Real, Categorical, Integer
from trill.utils.logging import setup_logger
from loguru import logger

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

# From Sean R Johnson at https://github.com/seanrjohnson/esmologs


#Importing required libraries
# import string
# import torch
# from torch.nn.utils.rnn import pad_sequence
# import torch.nn.functional as F
# from tqdm import tqdm
# from Bio import SeqIO
# import numpy as np
# import importlib.resources as importlib_resources

# def _open_if_is_name(filename_or_handle, mode="r"):
#     """
#         if a file handle is passed, return the file handle
#         if a Path object or path string is passed, open and return a file handle to the file.
#         returns:
#             file_handle, input_type ("name" | "handle")
#     """
#     out = filename_or_handle
#     input_type = "handle"
#     try:
#         out = open(filename_or_handle, mode)
#         input_type = "name"
#     except TypeError:
#         pass
#     except Exception as e:
#         raise(e)

#     return (out, input_type)


# def get_data():
#     pkg = importlib_resources.files("threedifam")
#     train_set_pep = list(SeqIO.parse(str(pkg / "data" / "train_set_pep.fasta"), "fasta"))
#     train_set_3di = list(SeqIO.parse(str(pkg / "data" / "train_set_3di.fasta"), "fasta"))
#     test_set_pep = list(SeqIO.parse(str(pkg / "data" / "test_set_pep.fasta"), "fasta"))
#     test_set_3di =list(SeqIO.parse(str(pkg / "data" / "test_set_3di.fasta"), "fasta"))
#     val_set_pep = list(SeqIO.parse(str(pkg / "data" / "val_set_pep.fasta"), "fasta"))
#     val_set_3di = list(SeqIO.parse(str(pkg / "data" / "val_set_3di.fasta"), "fasta"))
    
#     #get only the sequences 
#     train_set_pep_seqs = parse_seqs_list(train_set_pep)
#     train_set_3di_seqs = parse_seqs_list(train_set_3di)
#     test_set_pep_seqs = parse_seqs_list(test_set_pep)
#     test_set_3di_seqs = parse_seqs_list(test_set_3di)
#     val_set_pep_seqs = parse_seqs_list(val_set_pep)
#     val_set_3di_seqs = parse_seqs_list(val_set_3di)
    
#     return train_set_pep_seqs,train_set_3di_seqs,test_set_pep_seqs,test_set_3di_seqs,val_set_pep_seqs,val_set_3di_seqs

# def parse_seqs_list(seqs_list):
#     seqs = []
#     #get a list of sequences
#     for record in seqs_list:
#         seqs.append(str(record.seq))
        
#     return seqs
             
# def seq2onehot(seq_records):
#     #declaring the alphabet
#     # - represents padding
#     alphabet = 'ACDEFGHIKLMNPQRSTVWYBXZJUO-'

#     # define a mapping of chars to integers
#     aa_to_int = dict((c, i) for i, c in enumerate(alphabet))
#     int_to_aa = dict((i, c) for i, c in enumerate(alphabet))
    
#     one_hot_representations = []
#     integer_encoded_representations = []

#     for seq in seq_records:
#         # integer encode seq data
#         integer_encoded = torch.tensor(np.array([aa_to_int[aa] for aa in seq]))
#         # convert tensors into one-hot encoding 
#         one_hot_representations.append(F.one_hot(integer_encoded, num_classes=27))

#     ps = pad_sequence(one_hot_representations,batch_first=True)
#     output = torch.transpose(ps,1,2)
#     return output

# def seq2integer(seq_records):
#     #declaring the alphabet
#     # - represents padding
#     alphabet = 'ACDEFGHIKLMNPQRSTVWYBXZJUO-'

#     # define a mapping of chars to integers
#     aa_to_int = dict((c, i) for i, c in enumerate(alphabet))
#     int_to_aa = dict((i, c) for i, c in enumerate(alphabet))
    
#     integer_encoded_representations = []

#     for seq in seq_records:
#         # integer encode seq data
#         integer_encoded = torch.tensor([aa_to_int[aa] for aa in seq])
#         integer_encoded_representations.append(integer_encoded)
    
#     #pad result to get equal sized tensors
#     output = pad_sequence(integer_encoded_representations,batch_first=True,padding_value=26.0)
#     return output

# class CleanSeq():
#     def __init__(self, clean=None):
#         self.clean = clean
#         if clean == 'delete':
#             # uses code from: https://github.com/facebookresearch/esm/blob/master/examples/contact_prediction.ipynb
#             deletekeys = dict.fromkeys(string.ascii_lowercase)
#             deletekeys["."] = None
#             deletekeys["*"] = None
#             translation = str.maketrans(deletekeys)
#             self.remove_insertions = lambda x: x.translate(translation)
#         elif clean == 'upper':
#             deletekeys = {'*': None, ".": "-"}
#             translation = str.maketrans(deletekeys)
#             self.remove_insertions = lambda x: x.upper().translate(translation)
            

#         elif clean == 'unalign':
#             deletekeys = {'*': None, ".": None, "-": None}
            
#             translation = str.maketrans(deletekeys)
#             self.remove_insertions = lambda x: x.upper().translate(translation)
        
#         elif clean is None:
#             self.remove_insertions = lambda x: x
        
#         else:
#             raise ValueError(f"unrecognized input for clean parameter: {clean}")
        
#     def __call__(self, seq):
#         return self.remove_insertions(seq)

#     def __repr__(self):
#         return f"CleanSeq(clean={self.clean})"




# def parse_fasta(filename, return_names=False, clean=None, full_name=False): 
#     """
#         adapted from: https://bitbucket.org/seanrjohnson/srj_chembiolib/src/master/parsers.py
        
#         input:
#             filename: the name of a fasta file or a filehandle to a fasta file.
#             return_names: if True then return two lists: (names, sequences), otherwise just return list of sequences
#             clean: {None, 'upper', 'delete', 'unalign'}
#                     if 'delete' then delete all lowercase "." and "*" characters. This is usually if the input is an a2m file and you don't want to preserve the original length.
#                     if 'upper' then delete "*" characters, convert lowercase to upper case, and "." to "-"
#                     if 'unalign' then convert to upper, delete ".", "*", "-"
#             full_name: if True, then returns the entire name. By default only the part before the first whitespace is returned.
#         output: sequences or (names, sequences)
#     """
    
#     prev_len = 0
#     prev_name = None
#     prev_seq = ""
#     out_seqs = list()
#     out_names = list()
#     (input_handle, input_type) = _open_if_is_name(filename)

#     seq_cleaner = CleanSeq(clean)

#     for line in input_handle:
#         line = line.strip()
#         if len(line) == 0:
#             continue
#         if line[0] == ">":
#             if full_name:
#                 name = line[1:]
#             else:
#                 parts = line.split(None, 1)
#                 name = parts[0][1:]
#             out_names.append(name)
#             if (prev_name is not None):
#                 out_seqs.append(prev_seq)
#             prev_len = 0
#             prev_name = name
#             prev_seq = ""
#         else:
#             prev_len += len(line)
#             prev_seq += line
#     if (prev_name != None):
#         out_seqs.append(prev_seq)

#     if input_type == "name":
#         input_handle.close()
    
#     if clean is not None:
#         for i in range(len(out_seqs)):
#             out_seqs[i] = seq_cleaner(out_seqs[i])

#     if return_names:
#         return out_names, out_seqs
#     else:
#         return out_seqs
    

# def iter_fasta(filename, clean=None, full_name=False): 
#     """
#         adapted from: https://bitbucket.org/seanrjohnson/srj_chembiolib/src/master/parsers.py
        
#         input:
#             filename: the name of a fasta file or a filehandle to a fasta file.
#             return_names: if True then return two lists: (names, sequences), otherwise just return list of sequences
#             clean: {None, 'upper', 'delete', 'unalign'}
#                     if 'delete' then delete all lowercase "." and "*" characters. This is usually if the input is an a2m file and you don't want to preserve the original length.
#                     if 'upper' then delete "*" characters, convert lowercase to upper case, and "." to "-"
#                     if 'unalign' then convert to upper, delete ".", "*", "-"
#             full_name: if True, then returns the entire name. By default only the part before the first whitespace is returned.
#         output: names, sequences
#     """
    
#     prev_len = 0
#     prev_name = None
#     prev_seq = ""
#     (input_handle, input_type) = _open_if_is_name(filename)

#     seq_cleaner = CleanSeq(clean)

#     for line in input_handle:
#         line = line.strip()
#         if len(line) == 0:
#             continue
#         if line[0] == ">":
#             if full_name:
#                 name = line[1:]
#             else:
#                 parts = line.split(None, 1)
#                 name = parts[0][1:]
#             if (prev_name is not None):
#                 yield prev_name, seq_cleaner(prev_seq)
#             prev_len = 0
#             prev_name = name
#             prev_seq = ""
#         else:
#             prev_len += len(line)
#             prev_seq += line
#     if (prev_name != None):
#         yield prev_name, seq_cleaner(prev_seq)
        

#     if input_type == "name":
#         input_handle.close()


# def fasta2foldseek(aa_fasta, tdi_fasta, output_basename):
#     # pep dbtype
#     with open(output_basename+".dbtype","wb") as pep_dbtype:
#         pep_dbtype.write(b'\x00\x00\x00\x00')

#     # 3Di dbtype
#     with open(output_basename+"_ss.dbtype","wb") as tdi_dbtype:
#         tdi_dbtype.write(b'\x00\x00\x00\x00')

#     # headers dbtype
#     with open(output_basename+"_h.dbtype","wb") as h_dbtype:
#         h_dbtype.write(b'\x00\x0c\x00\x00')
        
    
#     with open(f"{output_basename}","wb") as aa_h, open(f"{output_basename}_ss","wb") as tdi_h, open(f"{output_basename}_h","wb") as header_h, \
#          open(f"{output_basename}.index","wb") as aa_index_h, open(f"{output_basename}_ss.index","wb") as tdi_index_h, open(f"{output_basename}_h.index","wb") as header_index_h, \
#          open(f"{output_basename}.lookup","wb") as lookup_h, \
#          open(tdi_fasta, "r") as tdi_in, open(aa_fasta, "r") as pep_in:
#         tdi_iter = iter_fasta(tdi_in, full_name=True)

#         seq_index = -1
#         for pep_header, pep_seq in iter_fasta(pep_in, full_name=True):
#             pep_name = pep_header.split(' ')[0]
#             tdi_header, tdi_seq = next(tdi_iter)
#             tdi_name = tdi_header.split(' ')[0]
#             assert pep_header == tdi_header, f"Headers do not match: {pep_header} != {tdi_header}"
#             assert pep_name == tdi_name, f"Names do not match: {pep_name} != {tdi_name}"
#             # assert len(pep_seq) == len(tdi_seq), f"Sequences do not match in length: {len(pep_seq)} != {len(tdi_seq)}"

#             seq_index += 1
            
#             # write the pep sequence
#             aa_start_pos = aa_h.tell()
#             aa_index_h.write(f"{seq_index}\t{aa_start_pos}\t".encode())
#             aa_h.write(pep_seq.encode())
#             aa_h.write(b'\x0a\x00')
#             aa_index_h.write(f"{aa_h.tell() - aa_start_pos}\n".encode())

#             # write the tdi sequence
#             tdi_start_pos = tdi_h.tell()
#             tdi_index_h.write(f"{seq_index}\t{tdi_start_pos}\t".encode())
#             tdi_h.write(tdi_seq.encode())
#             tdi_h.write(b'\x0a\x00')
#             tdi_index_h.write(f"{tdi_h.tell() - tdi_start_pos}\n".encode())

#             # write the header
#             header_start_pos = header_h.tell()
#             header_index_h.write(f"{seq_index}\t{header_start_pos}\t".encode())
#             header_h.write(pep_header.encode())
#             header_h.write(b'\x0a\x00')
#             header_index_h.write(f"{header_h.tell() - header_start_pos}\n".encode())

#             # write the lookup
#             lookup_h.write(f"{seq_index}\t{pep_name}\t{seq_index}\n".encode())

# def prep_foldseek_dbs(fasta_aa, fasta_3di, output_basename):
#     # read in amino-acid sequences
#     sequences_aa = {}
#     for record in SeqIO.parse(fasta_aa, "fasta"):
#         sequences_aa[record.id] = str(record.seq)

#     # read in 3Di strings
#     sequences_3di = {}
#     for record in SeqIO.parse(fasta_3di, "fasta"):
#         if not record.id in sequences_aa.keys():
#             print("Warning: ignoring 3Di entry {}, since it is not in the amino-acid FASTA file".format(record.id))
#         else:
#             sequences_3di[record.id] = str(record.seq).upper()

#     # assert that we parsed 3Di strings for all sequences in the amino-acid FASTA file
#     for id in sequences_aa.keys():
#         if not id in sequences_3di.keys():
#             print("Error: entry {} in amino-acid FASTA file has no corresponding 3Di string".format(id))
#             quit()

#     # generate TSV file contents
#     tsv_aa = ""
#     tsv_3di = ""
#     tsv_header = ""
#     for i,id in enumerate(sequences_aa.keys()):
#         tsv_aa += "{}\t{}\n".format(str(i+1), sequences_aa[id])
#         tsv_3di += "{}\t{}\n".format(str(i+1), sequences_3di[id])
#         tsv_header += "{}\t{}\n".format(str(i+1), id)

#     # write TSV files
#     with open("aa.tsv", "w+") as f:
#         f.write(tsv_aa)
#     with open("3di.tsv", "w+") as f:
#         f.write(tsv_3di)
#     with open("header.tsv", "w+") as f:
#         f.write(tsv_header)

#     # create Foldseek database
#     os.system("foldseek tsv2db aa.tsv {} --output-dbtype 0".format(output_basename))
#     os.system("foldseek tsv2db 3di.tsv {}_ss --output-dbtype 0".format(output_basename))
#     os.system("foldseek tsv2db header.tsv {}_h --output-dbtype 12".format(output_basename))

    # clean up
    # os.remove("aa.tsv")
    # os.remove("3di.tsv")
    # os.remove("header.tsv")
#     # read in amino-acid sequences
#     sequences_aa = {}
#     for record in SeqIO.parse(fasta_aa, "fasta"):
#         sequences_aa[record.id] = str(record.seq)

#     # read in 3Di strings
#     sequences_3di = {}
#     for record in SeqIO.parse(fasta_3di, "fasta"):
#         sequences_3di[record.id] = str(record.seq).upper()

#     # generate TSV file contents
#     tsv_aa = ""
#     tsv_3di = ""
#     tsv_header = ""
#     for i, id in enumerate(sequences_aa.keys()):
#         tsv_aa += "{}\t{}\n".format(str(i + 1), sequences_aa[id])
#         tsv_3di += "{}\t{}\n".format(str(i + 1), sequences_3di[id])
#         tsv_header += "{}\t{}\n".format(str(i + 1), id)

#     with open(f'tmp_{output_basename}_aa.tsv', "w+") as f:
#         f.write(tsv_aa)
#     with open(f'tmp_{output_basename}_tdi.tsv', "w+") as f:
#         f.write(tsv_3di)
#     with open(f'tmp_{output_basename}_header.tsv', "w+") as f:
#         f.write(tsv_header)

#     for type, code in [('aa', 0), ('tdi', 0), ('header', 12)]:
#         ic(type)
#         ic(code)
#         foldseek_tsv2db = f'foldseek tsv2db tmp_{output_basename}_{type}.tsv tmp_{output_basename}_{type}_db --output-dbtype {code}'.split()
#         subprocess.run(foldseek_tsv2db)
#         print('eyee')