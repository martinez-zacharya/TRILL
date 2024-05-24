def setup(subparsers):
    fold = subparsers.add_parser(
        "fold",
        help="Predict 3D protein structures using ESMFold or obtain 3Di structure for use with Foldseek to perform "
             "remote homology detection")

    fold.add_argument(
        "model",
        help="Choose your desired model.",
        choices=("ESMFold", "ProstT5")
    )
    fold.add_argument(
        "--strategy",
        help="ESMFold: Choose a specific strategy if you are running out of CUDA memory. You can also pass either 64, "
             "or 32 for model.trunk.set_chunk_size(x)",
        action="store",
        default=None,
    )
    fold.add_argument(
        "--batch_size",
        help="ESMFold: Change batch-size number for folding proteins. Default is 1",
        action="store",
        default=1,
        dest="batch_size",
    )

    fold.add_argument(
        "query",
        help="Input fasta file",
        action="store"
    )


def process_sublist(sublist):
    if isinstance(sublist, tuple) and len(sublist) == 2:
        return [sublist]
    elif isinstance(sublist, list):
        return sublist
    else:
        print(f"Unexpected data structure: {sublist=}")
    return []


def run(args):
    import os

    import esm
    import pandas as pd
    import pytorch_lightning as pl
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, EsmForProteinFolding
    from loguru import logger
    from trill.utils.esm_utils import convert_outputs_to_pdb
    from trill.utils.lightning_models import CustomWriter, ProstT5
    # from trill.utils.rosettafold_aa import rfaa_setup
    from .commands_common import cache_dir, get_logger

    ml_logger = get_logger(args)

    if args.model == "ESMFold":
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        if int(args.GPUs) == 0:
            model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True,
                                                         torch_dtype="auto")
        else:
            model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", device_map="sequential",
                                                         torch_dtype="auto")
            model = model.cuda()
            model.esm = model.esm.half()
            model = model.cuda()
        if args.strategy is not None:
            model.trunk.set_chunk_size(int(args.strategy))
        fold_df = pd.DataFrame(list(data), columns=("Entry", "Sequence"))
        sequences = fold_df.Sequence.tolist()
        with torch.no_grad():
            for input_ids in tqdm(range(0, len(sequences), int(args.batch_size))):
                i = input_ids
                batch_input_ids = sequences[i: i + int(args.batch_size)]
                if int(args.GPUs) == 0:
                    if int(args.batch_size) > 1:
                        tokenized_input = tokenizer(batch_input_ids, return_tensors="pt",
                                                    add_special_tokens=False, padding=True)["input_ids"]
                    else:
                        tokenized_input = tokenizer(batch_input_ids, return_tensors="pt",
                                                    add_special_tokens=False)["input_ids"]
                    tokenized_input = tokenized_input.clone().detach()
                    prot_len = len(batch_input_ids[0])
                    try:
                        output = model(tokenized_input)
                        output = {key: val.cpu() for key, val in output.items()}
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"Protein too long to fold for current hardware: {prot_len} amino acids long)")
                            print(e)
                        else:
                            print(e)
                            pass
                else:
                    if int(args.batch_size) > 1:
                        tokenized_input = tokenizer(batch_input_ids, return_tensors="pt",
                                                    add_special_tokens=False, padding=True)["input_ids"]
                        prot_len = len(batch_input_ids[0])
                    else:
                        tokenized_input = tokenizer(batch_input_ids, return_tensors="pt",
                                                    add_special_tokens=False)["input_ids"]
                        prot_len = len(batch_input_ids[0])
                    tokenized_input = tokenized_input.clone().detach()
                    try:
                        tokenized_input = tokenized_input.to(model.device)
                        output = model(tokenized_input)
                        output = {key: val.cpu() for key, val in output.items()}
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"Protein too long to fold for current hardware: {prot_len} amino acids long)")
                            print(e)
                        else:
                            print(e)
                output = convert_outputs_to_pdb(output)
                if int(args.batch_size) > 1:
                    start_idx = i
                    end_idx = i + int(args.batch_size)
                    identifier = fold_df.Entry[start_idx:end_idx].tolist()
                else:
                    identifier = [fold_df.Entry[i]]
                for out, iden in zip(output, identifier):
                    with open(os.path.join(args.outdir, f"{iden}.pdb"), "w") as f:
                        f.write("".join(out))

    elif args.model == "ProstT5":
        model = ProstT5(args)
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=int(args.batch_size), num_workers=0)
        pred_writer = CustomWriter(output_dir=args.outdir, write_interval="epoch")
        if int(args.GPUs) == 0:
            trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], logger=ml_logger,
                                 num_nodes=int(args.nodes))
        else:
            trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), callbacks=[pred_writer],
                                 accelerator="gpu", logger=ml_logger, num_nodes=int(args.nodes))

        reps = trainer.predict(model, dataloader)
        cwd_files = os.listdir(args.outdir)
        pt_files = [file for file in cwd_files if "predictions_" in file]
        pred_embeddings = []
        if args.batch_size == 1 or int(args.GPUs) > 1:
            for pt in pt_files:
                preds = torch.load(os.path.join(args.outdir, pt))
                for pred in preds:
                    for sublist in pred:
                        if len(sublist) == 2 and args.batch_size == 1:
                            pred_embeddings.append(tuple([sublist[0], sublist[1]]))
                        else:
                            processed_sublists = process_sublist(sublist)
                            for sub in processed_sublists:
                                pred_embeddings.append(tuple([sub[0], sub[1]]))
            embedding_df = pd.DataFrame(pred_embeddings, columns=("3Di", "Label"))
            finaldf = embedding_df["3Di"].apply(pd.Series)
            finaldf["Label"] = embedding_df["Label"]
        else:
            embs = []
            for rep in reps:
                inner_embeddings = [item[0] for item in rep]
                inner_labels = [item[1] for item in rep]
                for emb_lab in zip(inner_embeddings, inner_labels):
                    embs.append(emb_lab)
            embedding_df = pd.DataFrame(embs, columns=("3Di", "Label"))
            finaldf = embedding_df["3Di"].apply(pd.Series)
            finaldf["Label"] = embedding_df["Label"]

        outname = os.path.join(args.outdir, f"{args.name}_{args.model}.csv")
        finaldf.to_csv(outname, index=False, header=("3Di", "Label"))
        for file in pt_files:
            os.remove(os.path.join(args.outdir, file))

    # elif args.model == "RFAA":
    #     rfaa_setup(args, cache_dir)
