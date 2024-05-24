def setup(subparsers):
    classify = subparsers.add_parser(
        "regress",
        help="Train you own regressors on input protein sequences and some sort of score.")

    classify.add_argument(
        "regressor",
        help="Train a custom regression model",
        choices=("Linear", "LightGBM")
    )
    classify.add_argument(
        "query",
        help="Fasta file of protein sequences",
        action="store"
    )
    classify.add_argument(
        "--key",
        help="Input a CSV, with your mappings for your embeddings where the first column is the label and the "
             "second column is the value.",
        action="store"
    )
    classify.add_argument(
        "--save_emb",
        help="Save csv of embeddings",
        action="store_true",
        default=False
    )
    classify.add_argument(
        "--emb_model",
        help="Select desired protein language model for embedding your query proteins to then train your custom "
             "classifier. Default is esm2_t12_35M",
        default="esm2_t12_35M",
        action="store",
        choices=("esm2_t6_8M", "esm2_t12_35M", "esm2_t30_150M", "esm2_t33_650M", "esm2_t36_3B", "esm2_t48_15B",
                 "ProtT5-XL", "ProstT5", "Ankh", "Ankh-Large")
    )
    classify.add_argument(
        "--train_split",
        help="Choose your train-test percentage split for training and evaluating your custom classifier. For "
             "example, --train .6 would split your input sequences into two groups, one with 60%% of the sequences to "
             "train and the other with 40%% for evaluating",
        action="store",
    )
    classify.add_argument(
        "--preTrained",
        help="Enter the path to your pre-trained XGBoost binary classifier that you've trained with TRILL. This will "
             "be a .json file.",
        action="store",
    )

    classify.add_argument(
        "--preComputed_Embs",
        help="Enter the path to your pre-computed embeddings. Make sure they match the --emb_model you select.",
        action="store",
        default=False
    )

    classify.add_argument(
        "--batch_size",
        help="Sets batch_size for embedding.",
        action="store",
        default=1
    )


    classify.add_argument(
        "--lr",
        help="LightGBM: Sets the learning rate. Default is 0.2",
        action="store",
        default=0.2
    )

    classify.add_argument(
        "--max_depth",
        help="LightGBM: Sets the maximum tree depth. Default is -1, no max tree depth.",
        action="store",
        default=-1
    )

    classify.add_argument(
        "--num_leaves",
        help="LightGBM: Sets the max number of leaves in one tree. Default is 31",
        action="store",
        default=31
    )

    classify.add_argument(
        "--n_estimators",
        help="LightGBM: Number of boosting rounds",
        action="store",
        default=115
    )
    # classify.add_argument(
    #     "--sweep",
    #     help="XGBoost/LightGBM: Use this flag to perform cross-validated bayesian optimization over the hyperparameter space.",
    #     action="store_true",
    #     default=False
    # )
    # classify.add_argument(
    #     "--sweep_cv",
    #     help="XGBoost/LightGBM: Change the number of folds used for cross-validation.",
    #     action="store",
    #     default=3
    # )
    # classify.add_argument(
    #     "--f1_avg_method",
    #     help="XGBoost/LightGBM: Change the scoring method used for calculated F1. Default is with no averaging.",
    #     action="store",
    #     default=None,
    #     choices=("macro", "weighted", "micro", "None")
    # )

    # classify.add_argument(
    #     "--epochs",
    #     help="ESM2+MLP: Set number of epochs to train ESM2+MLP classifier.",
    #     action="store",
    #     default=3
    # )

def run(args):
    import builtins
    import logging
    import os
    import shutil
    import subprocess
    import sys
    import skops.io as sio

    import esm
    import numpy as np
    import pandas as pd
    import pytorch_lightning as pl
    import skops.io as sio
    import torch
    import xgboost as xgb
    from Bio import SeqIO
    from git import Repo
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import LabelEncoder
    from tqdm import tqdm
    from loguru import logger 
    from icecream import ic
    from sklearn.metrics import precision_recall_fscore_support
    import trill.utils.ephod_utils as eu
    from trill.commands.fold import process_sublist
    from trill.utils.MLP import MLP_C2H2, inference_epoch
    from trill.utils.classify_utils import prep_data, log_results, sweep, prep_hf_data, custom_esm2mlp_test, train_model, load_model, custom_model_test, predict_and_evaluate
    from trill.utils.regression_utils import log_reg_results, train_reg_model, prep_reg_data, predict_and_evaluate_reg, load_reg_model, custom_model_reg_test
    from trill.utils.esm_utils import parse_and_save_all_predictions, convert_outputs_to_pdb
    from .commands_common import cache_dir, get_logger

    outfile = os.path.join(args.outdir, f"{args.name}_{args.regressor}.out")
    if not args.preComputed_Embs:
        embed_command = (
            "trill",
            args.name,
            args.GPUs,
            "--outdir", args.outdir,
            "embed",
            args.emb_model,
            args.query,
            "--avg",
            "--batch_size", args.batch_size
        )
        subprocess.run(embed_command, check=True)
        df = pd.read_csv(os.path.join(args.outdir, f"{args.name}_{args.emb_model}_AVG.csv"))
    else:
        df = pd.read_csv(args.preComputed_Embs)
    if args.train_split is not None:
        train_df, test_df = prep_reg_data(df, args)
        command_line_args = sys.argv
        command_line_str = " ".join(command_line_args)
        clf = train_reg_model(train_df, args)
        if args.regressor == 'LightGBM':
            # clf.booster_.save_model(os.path.join(args.outdir, f"{args.name}_LightGBM-Regression.json"))
            sio.dump(clf, os.path.join(args.outdir, f"{args.name}_LightGBM-Regression.skops"))
        elif args.regressor == 'Linear':
            sio.dump(clf, os.path.join(args.outdir, f"{args.name}_Linear-Regression.skops"))
        r2, rmse = predict_and_evaluate_reg(clf, test_df, args)
        log_reg_results(outfile, command_line_str, args, r2=r2, rmse=rmse)
    else:
        model = load_reg_model(args)
        custom_model_reg_test(model, df, args)
        if not args.save_emb and not args.preComputed_Embs:
            os.remove(os.path.join(args.outdir, f"{args.name}_{args.emb_model}_AVG.csv"))

    if not args.save_emb and not args.preComputed_Embs:
        os.remove(os.path.join(args.outdir, f"{args.name}_{args.emb_model}_AVG.csv"))