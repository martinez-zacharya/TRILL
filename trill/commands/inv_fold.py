def setup(subparsers):
    inv_fold = subparsers.add_parser("inv_fold_gen", help="Generate proteins using inverse folding")
    inv_fold.add_argument(
        "model",
        help="Select which model to generate proteins using inverse folding.",
        choices=("ESM-IF1", "ProteinMPNN", "ProstT5")
    )

    inv_fold.add_argument(
        "query",
        help="Input pdb file for inverse folding",
        action="store"
    )

    inv_fold.add_argument(
        "--temp",
        help="Choose sampling temperature.",
        action="store",
        default="1"
    )

    inv_fold.add_argument(
        "--num_return_sequences",
        help="Choose number of proteins to generate.",
        action="store",
        default=1
    )

    inv_fold.add_argument(
        "--max_length",
        help="Max length of proteins generated, default is 500 AAs",
        default=500,
        type=int
    )

    inv_fold.add_argument(
        "--top_p",
        help="ProstT5: If set to float < 1, only the smallest set of most probable tokens with probabilities that add "
             "up to top_p or higher are kept for generation. Default is 1",
        default=1
    )
    inv_fold.add_argument(
        "--repetition_penalty",
        help="ProstT5: The parameter for repetition penalty. 1.0 means no penalty, the default is 1.2",
        default=1.2
    )
    inv_fold.add_argument(
        "--dont_sample",
        help="ProstT5: By default, the model will sample to generate the protein. With this flag, you can enable "
             "greedy decoding, where the most probable tokens will be returned.",
        default=True,
        action="store_false"
    )
    inv_fold.add_argument("--mpnn_model", type=str, default="v_48_020",
                          help="ProteinMPNN: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges "
                               "0.10A noise")
    inv_fold.add_argument("--save_score", type=int, default=0,
                          help="ProteinMPNN: 0 for False, 1 for True; save score=-log_prob to npy files")
    inv_fold.add_argument("--save_probs", type=int, default=0,
                          help="ProteinMPNN: 0 for False, 1 for True; save MPNN predicted probabilites per position")
    inv_fold.add_argument("--score_only", type=int, default=0,
                          help="ProteinMPNN: 0 for False, 1 for True; score input backbone-sequence pairs")
    inv_fold.add_argument("--path_to_fasta", type=str, default="",
                          help="ProteinMPNN: score provided input sequence in a fasta format; e.g. GGGGGG/PPPPS/WWW "
                               "for chains A, B, C sorted alphabetically and separated by /")
    inv_fold.add_argument("--conditional_probs_only", type=int, default=0,
                          help="ProteinMPNN: 0 for False, 1 for True; output conditional probabilities p(s_i given "
                               "the rest of the sequence and backbone)")
    inv_fold.add_argument("--conditional_probs_only_backbone", type=int, default=0,
                          help="ProteinMPNN: 0 for False, 1 for True; if true output conditional probabilities p(s_i "
                               "given backbone)")
    inv_fold.add_argument("--unconditional_probs_only", type=int, default=0,
                          help="ProteinMPNN: 0 for False, 1 for True; output unconditional probabilities p(s_i given "
                               "backbone) in one forward pass")
    inv_fold.add_argument("--backbone_noise", type=float, default=0.00,
                          help="ProteinMPNN: Standard deviation of Gaussian noise to add to backbone atoms")
    inv_fold.add_argument("--batch_size", type=int, default=1,
                          help="ProteinMPNN: Batch size; can set higher for titan, quadro GPUs, reduce this if "
                               "running out of GPU memory")
    inv_fold.add_argument("--pdb_path_chains", type=str, default='',
                          help="ProteinMPNN: Define which chains need to be designed for a single PDB ")
    inv_fold.add_argument("--chain_id_jsonl", type=str, default='',
                          help="ProteinMPNN: Path to a dictionary specifying which chains need to be designed and "
                               "which ones are fixed, if not specied all chains will be designed.")
    inv_fold.add_argument("--fixed_positions_jsonl", type=str, default='',
                          help="ProteinMPNN: Path to a dictionary with fixed positions")
    inv_fold.add_argument("--omit_AAs", type=list, default='X',
                          help="ProteinMPNN: Specify which amino acids should be omitted in the generated sequence, "
                               "e.g. 'AC' would omit alanine and cystine.")
    inv_fold.add_argument("--bias_AA_jsonl", type=str, default='',
                          help="ProteinMPNN: Path to a dictionary which specifies AA composion bias if neededi, "
                               "e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
    inv_fold.add_argument("--bias_by_res_jsonl", default='',
                          help="ProteinMPNN: Path to dictionary with per position bias.")
    inv_fold.add_argument("--omit_AA_jsonl", type=str, default='',
                          help="ProteinMPNN: Path to a dictionary which specifies which amino acids need to be omited "
                               "from design at specific chain indices")
    inv_fold.add_argument("--pssm_jsonl", type=str, default='', help="ProteinMPNN: Path to a dictionary with pssm")
    inv_fold.add_argument("--pssm_multi", type=float, default=0.0,
                          help="ProteinMPNN: A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN "
                               "predictions")
    inv_fold.add_argument("--pssm_threshold", type=float, default=0.0,
                          help="ProteinMPNN: A value between -inf + inf to restric per position AAs")
    inv_fold.add_argument("--pssm_log_odds_flag", type=int, default=0, help="ProteinMPNN: 0 for False, 1 for True")
    inv_fold.add_argument("--pssm_bias_flag", type=int, default=0, help="ProteinMPNN: 0 for False, 1 for True")
    inv_fold.add_argument("--tied_positions_jsonl", type=str, default="",
                          help="ProteinMPNN: Path to a dictionary with tied positions")


def run(args, logger, profiler):
    import os
    import shutil
    import subprocess
    import sys

    import pytorch_lightning as pl
    import torch
    from Bio import PDB
    from git import Repo

    from trill.utils.esm_utils import ESM_IF1_Wrangle, ESM_IF1
    from trill.utils.lightning_models import ProstT5, Custom3DiDataset
    from .commands_common import cache_dir

    if args.model == "ESM-IF1":
        if args.query is None:
            raise Exception("A PDB or CIF file is needed for generating new proteins with ESM-IF1")
        data = ESM_IF1_Wrangle(args.query)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        sample_df, native_seq_df = ESM_IF1(dataloader, genIters=int(args.num_return_sequences), temp=float(args.temp),
                                           GPUs=int(args.GPUs))
        pdb_name = args.query.split(".")[-2].split("/")[-1]
        with open(os.path.join(args.outdir, f"{args.name}_ESM-IF1_gen.fasta"), "w+") as fasta:
            for ix, row in native_seq_df.iterrows():
                fasta.write(f">{pdb_name}_chain-{row[1]} \n")
                fasta.write(f"{row[0][0]}\n")
            for ix, row in sample_df.iterrows():
                fasta.write(f">{args.name}_ESM-IF1_chain-{row[1]} \n")
                fasta.write(f"{row[0]}\n")
    elif args.model == "ProteinMPNN":
        if not os.path.exists((os.path.join(cache_dir, "ProteinMPNN/"))):
            print("Cloning forked ProteinMPNN")
            os.makedirs(os.path.join(cache_dir, "ProteinMPNN/"))
            proteinmpnn = Repo.clone_from("https://github.com/martinez-zacharya/ProteinMPNN",
                                          (os.path.join(cache_dir, "ProteinMPNN/")))
            mpnn_git_root = proteinmpnn.git.rev_parse("--show-toplevel")
            subprocess.run(("pip", "install", "-e", mpnn_git_root))
            sys.path.insert(0, (os.path.join(cache_dir, "ProteinMPNN/")))
        else:
            sys.path.insert(0, (os.path.join(cache_dir, "ProteinMPNN/")))
        from mpnnrun import run_mpnn
        print("ProteinMPNN generation starting...")
        run_mpnn(args)

    elif args.model == "ProstT5":
        model = ProstT5(args)
        os.makedirs('foldseek_intermediates')
        create_db_cmd = f'foldseek createdb {os.path.abspath(args.query)} DB'.split()
        subprocess.run(create_db_cmd, cwd='foldseek_intermediates')
        lndb_cmd = f'foldseek lndb DB_h DB_ss_h'.split()
        subprocess.run(lndb_cmd, cwd='foldseek_intermediates')
        convert_cmd = (f'foldseek convert2fasta foldseek_intermediates/DB_ss {os.path.join(args.outdir, args.name)}_ss'
                       f'.3di').split()
        subprocess.run(convert_cmd)
        shutil.rmtree("foldseek_intermediates")

        data = Custom3DiDataset(f"{os.path.join(args.outdir, args.name)}_ss.3di")
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        chain_id_list = []
        pdb_parser = PDB.PDBParser(QUIET=True)
        pdb = pdb_parser.get_structure("NA", args.query)
        for x in pdb:
            for chain in x:
                chain_id_list.append(chain.id)
        if int(args.GPUs) == 0:
            trainer = pl.Trainer(enable_checkpointing=False, logger=logger, num_nodes=int(args.nodes))
        else:
            trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), accelerator="gpu", logger=logger,
                                 num_nodes=int(args.nodes))
        with open(os.path.join(args.outdir, f"{args.name}_ProstT5_InvFold.fasta"), "w+") as fasta:
            for i in range(int(args.num_return_sequences)):

                out = trainer.predict(model, dataloader)
                for seq, chain_id in zip(out, chain_id_list):
                    fasta.write(f">{args.name}_ProstT5_InvFold_Chain-{chain_id}_{i} \n")
                    fasta.write(f"{seq}\n")
                fasta.flush()
