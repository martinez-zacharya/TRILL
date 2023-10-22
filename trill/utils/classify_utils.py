import argparse
import os
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
import xgboost as xgb
import multiprocessing
import pandas as pd
import os
import sys
import subprocess
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_data(args):
    if not args.preComputed_Embs:
        embed_command = f"trill {args.name} {args.GPUs} --outdir {args.outdir} embed {args.emb_model} {args.query} --avg"
        subprocess.run(embed_command.split(' '), check=True)
        df = pd.read_csv(os.path.join(args.outdir, f'{args.name}_{args.emb_model}_AVG.csv'))
    else:
        df = pd.read_csv(args.preComputed_Embs)
    return df

def prep_data(df, args):
    if args.train_split is not None:
        if args.key == None:
            raise Exception('Training an XGBoost classifier requires a class key CSV!')
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

def train_model(train_df, args):
    model = xgb.XGBClassifier(objective = 'multi:softprob', num_class=train_df['NewLab'].nunique(), gamma = args.xg_gamma, learning_rate = args.xg_lr,  max_depth = args.xg_max_depth, n_estimators = args.n_estimators, reg_alpha = args.xg_reg_alpha, reg_lambda = args.xg_reg_lambda, random_state = args.RNG_seed)
    model.fit(train_df.iloc[:, :-2], train_df['NewLab'])
    return model

def xg_test(model, le, test_df, args):
    if args.f1_avg_method == 'None':
        args.f1_avg_method = None
    test_preds_proba = model.predict_proba(test_df.iloc[:, :-2])
    proba_df = pd.DataFrame(test_preds_proba, columns=le.inverse_transform(range(len(test_df['NewLab'].unique()))))
    test_preds = proba_df.idxmax(axis=1)
    proba_df['Label'] = test_df['Label'].values
    proba_df.to_csv(os.path.join(args.outdir, f'{args.name}_XGBoost_class_probs.csv'), index=False)
    transformed_preds = le.transform(test_preds)
    zipped = zip(test_preds, test_df["Label"])
    pred_df = pd.DataFrame(zipped)
    # pred_df['Label'] = test_df['Label']
    pred_df.to_csv(os.path.join(args.outdir, f'{args.name}_XGBoost_predictions.csv'), header=['Prediction', 'Label'],index=False)
    transformed_preds = transformed_preds.astype(test_df['NewLab'].dtype)
    precision, recall, fscore, support = precision_recall_fscore_support(test_df['NewLab'].values, transformed_preds, average=args.f1_avg_method, labels=np.unique(test_df['NewLab']))
    return precision, recall, fscore, support

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

def log_results(out_file, command_str, n_classes, args, classes = None, sweeped_clf=None, precision=None, recall=None, fscore=None, support=None):
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
                out.write(f"\nClass: {label}\n")
                
                out.write(f"\tPrecision: {precision[i]}\n")
                
                out.write(f"\tRecall: {recall[i]}\n")
                
                out.write(f"\tF-score: {fscore[i]}\n")

                out.write(f"\tSupport: {support[i]}\n")
        elif precision is not None and recall is not None and fscore is not None and support is not None:  
            out.write("Classification Metrics Per Class:\n")
            for i, label in enumerate(classes):
                out.write(f"\nClass: {label}\n")
                
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
                    print(f"File {path} does not exist.")
                    continue
                
                with open(path, 'r') as fasta_file:
                    for line in fasta_file:
                        line = line.strip()
                        if line.startswith('>'):
                            all_headers.append(line[1:])
                            all_labels.append(class_label)
    else:
        print('prepare_class_key requires either a path to a directory of fastas or a text file of fasta paths!')
        raise RuntimeError
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'Label': all_headers,
        'Class': all_labels
    })
    outpath = os.path.join(args.outdir, f'{args.name}_class_key.csv')
    df.to_csv(outpath, index=False)
    print(f"Class key CSV generated and saved as '{outpath}'.")

def sweep(train_df, args):
    np.int = np.int64
    if args.n_workers == 1:
        print("WARNING!")
        print("You are trying to perform a hyperparameter sweep with only 1 core! You can substantially speed up the sweep by increasing this number to the amount of CPU cores available!")
        print(f"In your case, you have {multiprocessing.cpu_count()} CPU cores available!")
    if train_df['NewLab'].nunique() == 2:
        model = xgb.XGBClassifier(objective='binary:logitraw')
        f1_avg_method = 'binary'
    else:
        model = xgb.XGBClassifier(objective='multi:softprob')
        f1_avg_method = 'macro'
    f1_scorer = make_scorer(f1_score, average=f1_avg_method)
    
    param_grid = {
        'booster': ['gbtree'],
        'gamma': [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25, 50, 100, 200],
        'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7],
        'max_depth': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'n_estimators': [50, 65, 80, 100, 115, 130, 150],
        'reg_alpha': [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25, 50, 100, 200],
        'reg_lambda': [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25, 50, 100, 200],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
    }

    clf = BayesSearchCV(estimator=model, search_spaces=param_grid, n_points=10, n_jobs=int(args.n_workers), 
                        scoring=f1_scorer, cv=args.sweep_cv, return_train_score=True, verbose=1)
    print("Sweeping...")
    clf.fit(train_df.iloc[:, :-2], train_df['NewLab'])
    clf.best_estimator_.save_model(os.path.join(args.outdir, f'{args.name}_XGBoost_{len(train_df.columns)-2}.json'))
    print("Sweep Complete! Now evaluating...")

    
    return clf

