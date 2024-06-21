def setup(subparsers):
    classify = subparsers.add_parser(
        "classify",
        help="Classify proteins using either pretrained classifiers or train/test your own.")

    classify.add_argument(
        "classifier",
        help="Predict thermostability/optimal enzymatic pH using TemStaPro/EpHod or choose custom to train/use your "
             "own XGBoost, LightGBM or Isolation Forest classifier. ESM2+MLP allows you to train an ESM2 model with a classification head end-to-end.",
        choices=("TemStaPro", "EpHod", "ECPICK", "XGBoost", "LightGBM", "iForest", "ESM2+MLP", "3Di-Search")
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
        help="Enter the path to your pre-trained classifier that you've trained with TRILL. This will "
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
        "--lr",
        help="XGBoost/LightGBM/ESM2+MLP: Sets the learning rate. Default is 0.0001 for ESM2+MLP, 0.2 for XGBoost and LightGBM",
        action="store",
        default=0.2
    )

    classify.add_argument(
        "--max_depth",
        help="XGBoost/LightGBM: Sets the maximum tree depth",
        action="store",
        default=8
    )

    classify.add_argument(
        "--num_leaves",
        help="LightGBM: Sets the max number of leaves in one tree. Default is 31",
        action="store",
        default=31
    )

    classify.add_argument(
        "--bagging_freq",
        help="LightGBM: Int that allows for bagging, which enables random sampling of training data of traingin data. For example, if it is set to 3, LightGBM will randomly sample the --bagging_frac of the data every 3rd iteration. Default is 0",
        action="store",
        default=0
    )

    classify.add_argument(
        "--bagging_frac",
        help="LightGBM: Sets fraction of training data to be used when bagging. Must be 0 < --bagging_frac <= 1. Default is 1",
        action="store",
        default=1
    )

    classify.add_argument(
        "--feature_frac",
        help="LightGBM: Sets fraction of training features to be randomly sampled for use in training. Must be 0 < --feature_frac <= 1. Default is 1",
        action="store",
        default=1
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
        help="XGBoost/LightGBM: Number of boosting rounds",
        action="store",
        default=115
    )
    classify.add_argument(
        "--sweep",
        help="XGBoost/LightGBM: Use this flag to perform cross-validated bayesian optimization over the hyperparameter space.",
        action="store_true",
        default=False
    )
    classify.add_argument(
        "--sweep_cv",
        help="XGBoost/LightGBM: Change the number of folds used for cross-validation.",
        action="store",
        default=3
    )
    classify.add_argument(
        "--f1_avg_method",
        help="XGBoost/LightGBM: Change the scoring method used for calculated F1. Default is with no averaging.",
        action="store",
        default=None,
        choices=("macro", "weighted", "micro", "None")
    )

    classify.add_argument(
        "--epochs",
        help="ESM2+MLP: Set number of epochs to train ESM2+MLP classifier.",
        action="store",
        default=3
    )

    classify.add_argument(
        "--db",
        help="3Di-Search: Specify the path of the fasta file for your database that you want to query against.",
        action="store",
    )

def run(args):
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
    from loguru import logger 
    from icecream import ic
    from sklearn.metrics import precision_recall_fscore_support
    import trill.utils.ephod_utils as eu
    from trill.commands.fold import process_sublist
    from trill.utils.MLP import MLP_C2H2, inference_epoch
    from trill.utils.classify_utils import prep_data, setup_esm2_hf, prep_foldseek_dbs, get_3di_embeddings, log_results, sweep, prep_hf_data, custom_esm2mlp_test, train_model, load_model, custom_model_test, predict_and_evaluate
    from trill.utils.esm_utils import parse_and_save_all_predictions, convert_outputs_to_pdb
    from trill.utils.lightning_models import ProtT5, CustomWriter, ProstT5
    from ecpick import ECPICK
    from .commands_common import cache_dir, get_logger

    ml_logger = get_logger(args)

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]
    if args.sweep and not args.train_split:
        logger.error("You need to provide a train-test fraction with --train_split!")
        raise Exception("You need to provide a train-test fraction with --train_split!")
    
    if args.classifier == 'ECPICK':
        args.cache_dir = cache_dir
        ecpick = ECPICK(args)
        ecpick.predict_fasta(fasta_path=args.query, output_path=args.outdir, args=args)

    if args.classifier == "TemStaPro":
        if not args.preComputed_Embs:
            data = esm.data.FastaBatchedDataset.from_file(args.query)
            model = ProtT5(args)
            pred_writer = CustomWriter(output_dir=args.outdir, write_interval="epoch")
            dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=1, num_workers=0)
            if int(args.GPUs) > 0:
                trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], devices=int(args.GPUs), accelerator="gpu",
                                     logger=ml_logger, num_nodes=int(args.nodes))
            else:
                trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], logger=ml_logger, num_nodes=int(args.nodes))
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
                clf.load_state_dict(torch.load(os.path.join(
                    temstapro_models_root, f"mean_major_imbal-{thresh}_s{seed}.pt")))
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
        inference_df = inference_df[["Protein", "Threshold", "Mean_Pred", "Binary_Pred"]]
        inference_df.to_csv(os.path.join(args.outdir, f"{args.name}_TemStaPro_preds.csv"), index=False)
        if not args.save_emb and not args.preComputed_Embs:
            os.remove(os.path.join(args.outdir, f"{args.name}_ProtT5_AVG.csv"))

    elif args.classifier == "EpHod":
        logging.getLogger("pytorch_lightning.utilities.rank_zero").addHandler(logging.NullHandler())
        logging.getLogger("pytorch_lightning.accelerators.cuda").addHandler(logging.NullHandler())
        if not os.path.exists(os.path.join(cache_dir, "EpHod_Models")):
            logger.info("Downloading EpHod models...")
            cmd = ("curl", "-o", "saved_models.tar.gz", "--progress-bar",
                   "https://zenodo.org/records/8011249/files/saved_models.tar.gz?download=1")
            result = subprocess.run(cmd)

            shutil.move("saved_models.tar.gz", cache_dir)
            tarfile = os.path.join(cache_dir, "saved_models.tar.gz")
            shutil.unpack_archive(tarfile, cache_dir)
            os.remove(tarfile)
            shutil.move(os.path.join(cache_dir, "saved_models"), os.path.join(cache_dir, "EpHod_Models"))

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
            logger.warning(warning)
            locs = np.argwhere(lengths <= 1022).flatten()
            headers, sequences, accessions = [array[locs] for array in (headers, sequences, accessions)]
            numseqs = len(sequences)

        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        # Prediction output file
        phout_file = os.path.join(args.outdir, f"{args.name}_EpHod.csv")
        embed_file = os.path.join(args.outdir, f"{args.name}_ESM1v_embeddings.csv")
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

    elif args.classifier == 'ESM2+MLP':
        if not args.preTrained:
            command_line_args = sys.argv
            command_line_str = " ".join(command_line_args)
            outfile = os.path.join(args.outdir, f"{args.name}_{args.classifier}_{args.emb_model}.out")
            logger.info("Prepping data for training ESM2+MLP...")
            train_df, test_df, n_classes, le = prep_hf_data(args)
            logger.info("Setting up ESM2+MLP...")
            trainer, test_dataset = setup_esm2_hf(train_df, test_df, args, n_classes)
            train_res = trainer.train()
            test_res = trainer.predict(test_dataset = test_dataset)
            trainer.model.save_pretrained(os.path.join(args.outdir, f'{args.name}_{args.emb_model}-MLP_{n_classes}-classifier.pt'), safe_serialization=False) 
            preds = np.argmax(test_res[0], axis=1)
            transformed_preds = le.inverse_transform(preds)
            unique_c = np.unique(transformed_preds)
            precision, recall, fscore, support = precision_recall_fscore_support(test_df['NewLab'].values, preds, average=args.f1_avg_method, labels=np.unique(test_df['NewLab']))
            log_results(outfile, command_line_str, n_classes, args, classes=unique_c, precision=precision, recall=recall, fscore=fscore, support=support, le=le)
        else:
            trainer, dataset, label_list = custom_esm2mlp_test(args)
            test_res = trainer.predict(test_dataset=dataset)
            # Convert probabilities to a DataFrame
            proba_df = pd.DataFrame(test_res[0])
            
            # Find the index of the maximum probability for each row (prediction)
            test_preds = proba_df.idxmax(axis=1)
            
            # Add the original labels to the DataFrame
            proba_df['Label'] = label_list
            
            # Save the probabilities to a CSV file
            proba_file_name = f'{args.name}_ESM2-MLP_class_probs.csv'
            proba_df.to_csv(os.path.join(args.outdir, proba_file_name), index=False)
            
            # Prepare and save the predictions to a CSV file
            pred_df = pd.DataFrame(test_preds, columns=['Prediction'])
            pred_df['Label'] = label_list
            
            pred_file_name = f'{args.name}_ESM2-MLP_predictions.csv'
            pred_df.to_csv(os.path.join(args.outdir, pred_file_name), index=False)


    elif args.classifier not in ['3Di-Search','ECPICK', 'iForest']:
        outfile = os.path.join(args.outdir, f"{args.name}_{args.classifier}.out")
        if not args.preComputed_Embs:
            embed_command = (
                "trill",
                args.name,
                args.GPUs,
                "--outdir", args.outdir,
                "embed",
                args.emb_model,
                args.query,
                "--avg"
            )
            subprocess.run(embed_command, check=True)
            df = pd.read_csv(os.path.join(args.outdir, f"{args.name}_{args.emb_model}_AVG.csv"))
        else:
            df = pd.read_csv(args.preComputed_Embs)

        if args.train_split is not None:
            le = LabelEncoder()
            train_df, test_df, n_classes = prep_data(df, args)
            unique_c = np.unique(test_df["NewLab"])
            classes = train_df["NewLab"].unique()
            train_df["NewLab"] = le.fit_transform(train_df["NewLab"])
            test_df["NewLab"] = le.transform(test_df["NewLab"])
            command_line_args = sys.argv
            command_line_str = " ".join(command_line_args)

            if args.sweep:
                sweeped_clf = sweep(train_df, args)
                precision, recall, fscore, support = predict_and_evaluate(sweeped_clf, le, test_df, args)
                log_results(outfile, command_line_str, n_classes, args, classes=unique_c, sweeped_clf=sweeped_clf,precision=precision, recall=recall, fscore=fscore, support=support, le=le)
            else:
                clf = train_model(train_df, args)
                clf.save_model(os.path.join(args.outdir, f"{args.name}_{args.classifier}_{len(train_df.columns) - 2}.json"))
                precision, recall, fscore, support = predict_and_evaluate(clf, le, test_df, args)
                log_results(outfile, command_line_str, n_classes, args, classes=classes, precision=precision,recall=recall, fscore=fscore, support=support, le=le)

            if not args.save_emb and not args.preComputed_Embs:
                os.remove(os.path.join(args.outdir, f"{args.name}_{args.emb_model}_AVG.csv"))

        elif args.classifier not in ['3Di-Search','ECPICK']:
            if not args.preTrained:
                logger.error("You need to provide a model with --preTrained to perform inference!")
                raise Exception("You need to provide a model with --preTrained to perform inference!")
            else:
                clf = load_model(args)
                custom_model_test(clf, df, args)

                if not args.save_emb and not args.preComputed_Embs:
                    os.remove(os.path.join(args.outdir, f"{args.name}_{args.emb_model}_AVG.csv"))

    elif args.classifier not in ['iForest','ECPICK']:
        # Load embeddings
        if not args.preComputed_Embs:
            embed_command = (
                "trill",
                args.name,
                args.GPUs,
                "--outdir", args.outdir,
                "embed",
                args.emb_model,
                args.query,
                "--avg"
            )
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


    elif args.classifier == '3Di-Search':
        logger.info(f'Prepping Foldseek databases from {args.query} and {args.db}')
        query_preds_out_path, db_preds_out_path = get_3di_embeddings(args, cache_dir)
        prep_foldseek_dbs(args.query, query_preds_out_path, f'{args.name}_query')
        prep_foldseek_dbs(args.db, db_preds_out_path, f'{args.name}_db')
        logger.info(f'Finished creating Foldseek databases!')
        foldseek_search_cmd = f'foldseek search tmp_{args.name}_query_db tmp_{args.name}_db_db {args.name}_3di-search_results tmp -v 0'
        logger.info(f'Starting foldseek search with: \n{foldseek_search_cmd}')
        subprocess.run(foldseek_search_cmd.split())
        output_path = os.path.join(args.outdir, f'{args.name}_3di-search_results.tsv')
        foldseek_convertalis_cmd = f'foldseek convertalis tmp_{args.name}_query_db tmp_{args.name}_db_db {args.name}_3di-search_results {output_path} -v 0'.split()
        subprocess.run(foldseek_convertalis_cmd)
        logger.info(f'Foldseek output can be found at {output_path}!')

