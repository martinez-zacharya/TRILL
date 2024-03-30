def setup(subparsers):
    classify = subparsers.add_parser(
        "classify",
        help="Classify proteins using either pretrained classifiers or train/test your own.")

    classify.add_argument(
        "classifier",
        help="Predict thermostability/optimal enzymatic pH using TemStaPro/EpHod or choose custom to train/use your "
             "own XGBoost or Isolation Forest classifier. Note for training XGBoost, you need to submit roughly equal "
             "amounts of each class as part of your query.",
        choices=("TemStaPro", "EpHod", "XGBoost", "iForest")
    )
    classify.add_argument(
        "query",
        help="Fasta file of sequences to score",
        action="store"
    )
    classify.add_argument(
        "--key",
        help="Input a CSV, with your class mappings for your embeddings where the first column is the label and the "
             "second column is the class.",
        action="store"
    )
    classify.add_argument(
        "--save_emb",
        help="Save csv of ProtT5 embeddings",
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
        help="EpHod: Sets batch_size for embedding with ESM1v.",
        action="store",
        default=1
    )

    classify.add_argument(
        "--xg_gamma",
        help="XGBoost: sets gamma for XGBoost, which is a hyperparameter that sets 'Minimum loss reduction required "
             "to make a further partition on a leaf node of the tree.'",
        action="store",
        default=0.4
    )

    classify.add_argument(
        "--xg_lr",
        help="XGBoost: Sets the learning rate for XGBoost",
        action="store",
        default=0.2
    )

    classify.add_argument(
        "--xg_max_depth",
        help="XGBoost: Sets the maximum tree depth",
        action="store",
        default=8
    )

    classify.add_argument(
        "--xg_reg_alpha",
        help="XGBoost: L1 regularization term on weights",
        action="store",
        default=0.8
    )

    classify.add_argument(
        "--xg_reg_lambda",
        help="XGBoost: L2 regularization term on weights",
        action="store",
        default=0.1
    )
    classify.add_argument(
        "--if_contamination",
        help="iForest: The amount of outliers in the data. Default is automatically determined, but you can set it "
             "between (0 , 0.5])",
        action="store",
        default="auto"
    )
    classify.add_argument(
        "--n_estimators",
        help="XGBoost/iForest: Number of boosting rounds",
        action="store",
        default=115
    )
    classify.add_argument(
        "--sweep",
        help="XGBoost: Use this flag to perform cross-validated bayesian optimization over the hyperparameter space.",
        action="store_true",
        default=False
    )
    classify.add_argument(
        "--sweep_cv",
        help="XGBoost: Change the number of folds used for cross-validation.",
        action="store",
        default=3
    )
    classify.add_argument(
        "--f1_avg_method",
        help="XGBoost: Change the scoring method used for calculated F1. Default is with no averaging.",
        action="store",
        default=None,
        choices=("macro", "weighted", "micro", "None")
    )


def run(args, logger, profiler):
    import builtins
    import logging
    import os
    import shutil
    import subprocess
    import sys

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

    import trill.utils.ephod_utils as eu
    from trill.utils.MLP import MLP_C2H2, inference_epoch
    from trill.utils.classify_utils import prep_data, log_results, xg_test, sweep, train_model, custom_xg_test
    from trill.utils.esm_utils import parse_and_save_all_predictions
    from trill.utils.lightning_models import ProtT5
    from .commands_common import cache_dir

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    if args.sweep and not args.train_split:
        raise Exception("You need to provide a train-test fraction with --train_split!")
    if args.classifier == "TemStaPro":
        if not args.preComputed_Embs:
            data = esm.data.FastaBatchedDataset.from_file(args.query)
            model = ProtT5(args)
            dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=1, num_workers=0)
            if int(args.GPUs) > 0:
                trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), accelerator="gpu",
                                     logger=logger, num_nodes=int(args.nodes))
            else:
                trainer = pl.Trainer(enable_checkpointing=False, logger=logger, num_nodes=int(args.nodes))
            reps = trainer.predict(model, dataloader)
            parse_and_save_all_predictions(args)
        if not os.path.exists(os.path.join(cache_dir, "TemStaPro_models")):
            temstapro_models = Repo.clone_from("https://github.com/martinez-zacharya/TemStaPro_models",
                                               os.path.join(cache_dir, "TemStaPro_models"))
            temstapro_models_root = temstapro_models.git.rev_parse("--show-toplevel")
        else:
            temstapro_models = Repo(os.path.join(cache_dir, "TemStaPro_models"))
            temstapro_models_root = temstapro_models.git.rev_parse("--show-toplevel")
        THRESHOLDS = ("40", "45", "50", "55", "60", "65")
        SEEDS = ("41", "42", "43", "44", "45")
        if not args.preComputed_Embs:
            emb_df = pd.read_csv(os.path.join(args.outdir, f"{args.name}_ProtT5_AVG.csv"))
        else:
            emb_df = pd.read_csv(args.preComputed_Embs)
        embs = emb_df[emb_df.columns[:-1]].applymap(lambda x: torch.tensor(x)).values.tolist()
        labels = emb_df.iloc[:, -1]
        list_of_tensors = [torch.tensor(l) for l in embs]
        input_data = list(zip(list_of_tensors, labels))
        custom_dataset = CustomDataset(input_data)
        emb_loader = torch.utils.data.DataLoader(custom_dataset, shuffle=False, batch_size=1, num_workers=0)
        inferences = {}
        for thresh in THRESHOLDS:
            threshold_inferences = {}
            for seed in SEEDS:
                clf = MLP_C2H2(1024, 512, 256)
                clf.load_state_dict(torch.load(f"{temstapro_models_root}/mean_major_imbal-{thresh}_s{seed}.pt"))
                clf.eval()
                if int(args.GPUs) > 0:
                    clf.to("cuda")
                    threshold_inferences[seed] = inference_epoch(clf, emb_loader, device="cuda")
                else:
                    threshold_inferences[seed] = inference_epoch(clf, emb_loader, device="cpu")
            for seq in threshold_inferences["41"].keys():
                mean_prediction = 0
                for seed in SEEDS:
                    mean_prediction += threshold_inferences[seed][seq]
                mean_prediction /= len(SEEDS)
                binary_pred = builtins.round(mean_prediction)
                inferences[f"{seq}$%#{thresh}"] = (mean_prediction, binary_pred)
        inference_df = pd.DataFrame.from_dict(inferences, orient="index", columns=("Mean_Pred", "Binary_Pred"))
        inference_df = inference_df.reset_index(names="RawLab")
        inference_df["Protein"] = inference_df["RawLab"].apply(lambda x: x.split("$%#")[0])
        inference_df["Threshold"] = inference_df["RawLab"].apply(lambda x: x.split("$%#")[-1])
        inference_df = inference_df.drop(columns="RawLab")
        inference_df = inference_df[("Protein", "Threshold", "Mean_Pred", "Binary_Pred")]
        inference_df.to_csv(os.path.join(args.outdir, f"{args.name}_TemStaPro_preds.csv"), index=False)
        if not args.save_emb:
            os.remove(os.path.join(args.outdir, f"{args.name}_ProtT5_AVG.csv"))

    elif args.classifier == "EpHod":
        logging.getLogger("pytorch_lightning.utilities.rank_zero").addHandler(logging.NullHandler())
        logging.getLogger("pytorch_lightning.accelerators.cuda").addHandler(logging.NullHandler())
        if not os.path.exists(os.path.join(cache_dir, "EpHod_Models")):
            cmd = ("curl", "-o", "saved_models.tar.gz", "--progress-bar",
                   "https://zenodo.org/record/8011249/files/saved_models.tar.gz?download=1")
            result = subprocess.run(cmd)

            shutil.move("saved_models.tar.gz", os.path.join(cache_dir, ""))
            tarfile = os.path.join(cache_dir, "saved_models.tar.gz")
            shutil.unpack_archive(tarfile)
            os.remove(tarfile)
        else:
            headers, sequences = eu.read_fasta(args.query)
            accessions = [head.split()[0] for head in headers]
            headers, sequences, accessions = [np.array(item) for item in (headers, sequences, accessions)]
            assert len(accessions) == len(headers) == len(sequences), "Fasta file has unequal headers and sequences"
            numseqs = len(sequences)

            # Check sequence lengths
            lengths = np.array([len(seq) for seq in sequences])
            long_count = np.sum(lengths > 1022)
            warning = f"{long_count} sequences are longer than 1022 residues and will be omitted"

            # Omit sequences longer than 1022
            if max(lengths) > 1022:
                print(warning)
                locs = np.argwhere(lengths <= 1022).flatten()
                headers, sequences, accessions = [array[locs] for array in (headers, sequences, accessions)]
                numseqs = len(sequences)

        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        # Prediction output file
        phout_file = f"{args.outdir}/{args.name}_EpHod.csv"
        embed_file = f"{args.outdir}/{args.name}_ESM1v_embeddings.csv"
        ephod_model = eu.EpHodModel(args)
        num_batches = int(np.ceil(numseqs / args.batch_size))
        all_ypred, all_emb_ephod = [], []
        batches = range(1, num_batches + 1)
        batches = tqdm(batches, desc="Predicting pHopt")
        for batch_step in batches:
            start_idx = (batch_step - 1) * args.batch_size
            stop_idx = batch_step * args.batch_size
            accs = accessions[start_idx: stop_idx]
            seqs = sequences[start_idx: stop_idx]

            # Predict with EpHod model
            ypred, emb_ephod, attention_weights = ephod_model.batch_predict(accs, seqs, args)
            all_ypred.extend(ypred.to("cpu").detach().numpy())
            all_emb_ephod.extend(emb_ephod.to("cpu").detach().numpy())

        if args.save_emb:
            all_emb_ephod = pd.DataFrame(np.array(all_emb_ephod), index=accessions)
            all_emb_ephod.to_csv(embed_file)

        all_ypred = pd.DataFrame(all_ypred, index=accessions, columns=["pHopt"])
        all_ypred = all_ypred.reset_index(drop=False)
        all_ypred.rename(columns={"index": "Label"}, inplace=True)
        all_ypred.to_csv(phout_file, index=False)


    elif args.classifier == "XGBoost":
        outfile = os.path.join(args.outdir, f"{args.name}_XGBoost.out")
        if not args.preComputed_Embs:
            embed_command = f"trill {args.name} {args.GPUs} --outdir {args.outdir} embed {args.emb_model} {args.query} --avg"
            subprocess.run(embed_command.split(" "), check=True)
            df = pd.read_csv(os.path.join(args.outdir, f"{args.name}_{args.emb_model}_AVG.csv"))
        else:
            df = pd.read_csv(args.preComputed_Embs)

        if args.train_split is not None:
            le = LabelEncoder()
            # print(df)
            train_df, test_df, n_classes = prep_data(df, args)
            unique_c = np.unique(test_df["NewLab"])
            classes = train_df["NewLab"].unique()
            train_df["NewLab"] = le.fit_transform(train_df["NewLab"])
            test_df["NewLab"] = le.transform(test_df["NewLab"])
            command_line_args = sys.argv
            command_line_str = " ".join(command_line_args)

            if args.sweep:
                sweeped_clf = sweep(train_df, args)
                precision, recall, fscore, support = xg_test(sweeped_clf, le, test_df, args)
                log_results(outfile, command_line_str, n_classes, args, classes=unique_c, sweeped_clf=sweeped_clf,
                            precision=precision, recall=recall, fscore=fscore, support=support)
            else:
                clf = train_model(train_df, args)
                clf.save_model(os.path.join(args.outdir, f"{args.name}_XGBoost_{len(train_df.columns) - 2}.json"))
                precision, recall, fscore, support = xg_test(clf, le, test_df, args)
                log_results(outfile, command_line_str, n_classes, args, classes=classes, precision=precision,
                            recall=recall, fscore=fscore, support=support)

            if not args.save_emb and not args.preComputed_Embs:
                os.remove(os.path.join(args.outdir, f"{args.name}_{args.emb_model}_AVG.csv"))

        else:
            if not args.preTrained:
                raise Exception("You need to provide a model with --args.preTrained to perform inference!")
            else:
                clf = xgb.XGBClassifier()
                clf.load_model(args.preTrained)
                custom_xg_test(clf, df, args)

                if not args.save_emb and not args.preComputed_Embs:
                    os.remove(os.path.join(args.outdir, f"{args.name}_{args.emb_model}_AVG.csv"))

    elif args.classifier == "iForest":
        # Load embeddings
        if not args.preComputed_Embs:
            embed_command = f"trill {args.name} {args.GPUs} --outdir {args.outdir} embed {args.emb_model} {args.query} --avg".split(
                " ")
            subprocess.run(embed_command, check=True)
            df = pd.read_csv(os.path.join(args.outdir, f"{args.name}_{args.emb_model}_AVG.csv"))
        else:
            df = pd.read_csv(args.preComputed_Embs)

        # Filter fasta file
        if args.preComputed_Embs and not args.preTrained:
            valid_labels = set(df["Label"])
            filtered_records_labels = {record.id for record in SeqIO.parse(args.query, "fasta") if
                                       record.id in valid_labels}
            df = df[df["Label"].isin(filtered_records_labels)]

        # Train or load model
        if not args.preTrained:
            model = IsolationForest(
                random_state=int(args.RNG_seed),
                verbose=True,
                n_estimators=int(args.n_estimators),
                contamination=float(args.if_contamination) if args.if_contamination != "auto" else "auto"
            )
            model.fit(df.iloc[:, :-1])
            sio.dump(model, os.path.join(args.outdir, f"{args.name}_iForest.skops"))
        else:
            model = sio.load(args.preTrained, trusted=True)

            # Predict and output results
            preds = model.predict(df.iloc[:, :-1])
            # unique_values, counts = np.unique(preds, return_counts=True)
            # for value, count in zip(unique_values, counts):
            #     print(f"{value}: {count}")

            df["Predicted_Class"] = preds
            out_df = df[("Label", "Predicted_Class")]
            out_df.to_csv(os.path.join(args.outdir, f"{args.name}_iForest_predictions.csv"), index=False)
        if not args.save_emb and not args.preComputed_Embs:
            os.remove(os.path.join(args.outdir, f"{args.name}_{args.emb_model}_AVG.csv"))
