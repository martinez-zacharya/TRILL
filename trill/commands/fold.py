def setup(subparsers):
    fold = subparsers.add_parser(
        "fold",
        help="Predict monomeric 3D protein structures using ESMFold, protein complexes with ligands using Boltz-1/Chai-1, and obtain 3Di structure for use with Foldseek to perform "
             "remote homology detection")

    fold.add_argument(
        "model",
        help="Choose your desired model.",
        choices=("Chai-1", "Boltz-2", "ESMFold", "ProstT5")
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
        "--msa",
        help="Boltz-2/Chai-1: Use ColabFold Server to generate MSA for input and subsequent inference, may improve results",
        action="store_true",
        default=False,
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
    import subprocess
    import sys
    import numpy as np
    from tqdm import tqdm
    from pathlib import Path
    from transformers import AutoTokenizer, EsmForProteinFolding
    from loguru import logger
    from trill.utils.esm_utils import convert_outputs_to_pdb
    from trill.utils.dock_utils import create_init_file
    from trill.utils.lightning_models import CustomWriter, ProstT5
    from trill.utils.safe_load import safe_torch_load
    from git import Repo
    import re
    import textwrap
    # from trill.utils.rarefold import setup_rarefold
    # from trill.utils.rosettafold_aa import rfaa_setup
    from .commands_common import cache_dir, get_logger

    ml_logger = get_logger(args)

    if args.model == "ESMFold":
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        if int(args.GPUs) == 0:
            model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True,
                                                         torch_dtype="auto", use_safetensors=True)
        else:
            model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", device_map="sequential",
                                                         torch_dtype="auto", use_safetensors=True)
            # model = model.cuda()
            model.esm = model.esm.half()
            # model = model.cuda()
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
                        continue  # Skip to next iteration when there's an error
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
                        continue  # Skip to next iteration when there's an error
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

    # elif args.model == "RareFold":
    #     logger.info('Finding RareFold dependencies...')
    #     setup_rarefold(cache_dir)
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
                preds = safe_torch_load(os.path.join(args.outdir, pt))
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

    
    elif args.model == 'Chai-1':
        if not os.path.exists(os.path.join(cache_dir, "chai-lab")):
            logger.info("Cloning Chai-Lab")
            os.makedirs(os.path.join(cache_dir, "chai-lab"))
            chai = Repo.clone_from("https://github.com/chaidiscovery/chai-lab",
                                       os.path.join(cache_dir, "chai-lab"))
            chai_root = chai.git.rev_parse("--show-toplevel")
            sys.path.insert(0, os.path.join(cache_dir, "chai-lab"))
        else:
            sys.path.insert(0, os.path.join(cache_dir, "chai-lab"))

        from chai_lab.chai1 import run_inference

        candidates = run_inference(
            fasta_file=Path(args.query),
            output_dir=Path(f'{args.outdir}/{args.name}_Chai_output'),
            # 'default' setup
            num_trunk_recycles=3,
            num_diffn_timesteps=200,
            use_msa_server=True if args.msa else False,
            seed=int(args.RNG_seed),
            # device="cuda:0",
            use_esm_embeddings=True,
        )


        npz_dir = Path(f'{args.outdir}/{args.name}_Chai_output')
        npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith(".npz")]

        data = []

        # Iterate through the .npz files
        for file_path in npz_files:
            # Load the .npz file
            npz_data = np.load(file_path)
            
            # Dictionary to store row data
            row_data = {"file_name": os.path.basename(file_path)}
            
            # Extract data for each key
            for key in npz_data.keys():
                value = npz_data[key]
                
                # Handle different data shapes
                if value.shape == (1,):
                    # Scalar-like values
                    row_data[key] = value[0]
                elif value.shape == (1, 3):
                    # 1D arrays with 3 elements
                    row_data[key] = value.flatten().tolist()
                elif value.shape == (1, 3, 3):
                    # 2D arrays with 3x3 elements
                    row_data[key] = value.squeeze().tolist()
                else:
                    row_data[key] = value.tolist()  # General fallback
            
            # Append the row data to the list
            data.append(row_data)

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(npz_dir, f'{args.name}_Chai_scores.csv'), index=False)

    elif args.model == 'Boltz-2':

        if not os.path.exists(os.path.join(cache_dir, "boltz")):
            logger.info("Cloning Boltz")
            os.makedirs(os.path.join(cache_dir, "boltz"))
            boltz = Repo.clone_from("https://github.com/jwohlwend/boltz",
                                       os.path.join(cache_dir, "boltz"))
            boltz_root = boltz.git.rev_parse("--show-toplevel")
            sys.path.insert(0, os.path.join(cache_dir, "boltz/src"))
            prim_py = os.path.join(cache_dir, 'boltz/src/boltz/model/layers/triangular_attention/primitives.py')

            with open(prim_py, 'r') as file:
                lines = file.readlines()

            # Modify the file content
            new_lines = []
            for line in lines:
                # Comment out the specific lines
                if 'fa_is_installed = importlib.util.find_spec("flash_attn") is not None' in line or \
                'if fa_is_installed:' in line or \
                'from flash_attn.bert_padding import unpad_input' in line or \
                'from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func' in line:
                    new_lines.append(f"# {line}")
                else:
                    new_lines.append(line)
            with open(prim_py, 'w') as file:
                file.writelines(new_lines)

            boltz_main_path = os.path.join(cache_dir, 'boltz/src/boltz/main.py')

            # Read the file
            with open(boltz_main_path, "r") as file:
                lines = file.readlines()

            # Define patterns to identify CLI-related lines and the main block
            cli_patterns = [
                "@click.group",
                "@cli.command",
                "@click.argument",
                "@click.option",
                "cli()",
            ]
            main_block_pattern = "if __name__ == \"__main__\":"

            # Start processing the lines
            new_lines = []
            inside_multiline_block = False
            inside_main_block = False
            parentheses_count = 0  # Track open and close parentheses for multiline blocks

            for line in lines:
                stripped_line = line.strip()

                # Start of a multi-line CLI decorator (e.g., @click.option)
                if any(pattern in stripped_line for pattern in cli_patterns):
                    new_lines.append(f"# {line}")  # Comment out the starting line
                    if stripped_line.endswith("(") and not stripped_line.endswith(")"):
                        inside_multiline_block = True
                        parentheses_count = stripped_line.count("(") - stripped_line.count(")")
                    continue

                # Handle lines inside multi-line blocks
                if inside_multiline_block:
                    new_lines.append(f"# {line}")  # Comment out each line in the block
                    parentheses_count += stripped_line.count("(") - stripped_line.count(")")
                    if parentheses_count <= 0:  # Block ends when parentheses are balanced
                        inside_multiline_block = False
                    continue

                # Detect and comment out the main block
                if stripped_line.startswith(main_block_pattern):
                    new_lines.append(f"# {line}")  # Comment out the start of the main block
                    inside_main_block = True
                    continue

                if inside_main_block:
                    new_lines.append(f"# {line}")  # Comment out all lines in the main block
                    if stripped_line == "":  # Empty line indicates the block might have ended
                        inside_main_block = False
                    continue

                # Replace `click.echo` with `print`
                if "click.echo" in stripped_line:
                    new_lines.append(line.replace("click.echo", "print"))
                    continue

                # Keep all other lines unchanged
                new_lines.append(line)

            # Dedent the entire file to ensure consistent formatting
            dedented_code = textwrap.dedent("".join(new_lines))

            # Save the modified file back
            with open(boltz_main_path, "w") as file:
                file.write(dedented_code)
                
        else:
            boltz = Repo(os.path.join(cache_dir, "boltz"))
            boltz_root = boltz.git.rev_parse("--show-toplevel")
            sys.path.insert(0, os.path.join(cache_dir, "boltz/src"))

        directories = ['src/boltz']
        for directory in directories:
            create_init_file(os.path.join(boltz_root, directory))


        from boltz.main import predict as boltz_predict
        try:
            boltz_predict(args.query, args.outdir, seed = int(args.RNG_seed), use_msa_server = True if args.msa else False)
        except RuntimeError as e:
            if "Missing MSA's in input and --use_msa_server flag not set." in str(e):
                logger.error("Error: Missing MSA's in input and --msa flag not set.")
                logger.error(
                    "You can run single-sequence mode without an MSA by adding 'empty' "
                    "to the end of the FASTA header like '>A|protein|empty'."
                    ""
                )
                raise
            else:
                raise
