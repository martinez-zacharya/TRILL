def setup(subparsers):
    inv_fold_gen = subparsers.add_parser("inv_fold_gen", help="Generate proteins using inverse folding")
    inv_fold_gen.add_argument(
        "model",
        help="Select which model to generate proteins using inverse folding.",
        choices=("ESM-IF1", "ProstT5", "LigandMPNN",)
    )

    inv_fold_gen.add_argument(
        "query",
        help="Input pdb file for inverse folding",
        action="store"
    )

    inv_fold_gen.add_argument(
        "--temp",
        help="Choose sampling temperature.",
        action="store",
        default="1",
        type=float
    )

    inv_fold_gen.add_argument(
        "--num_return_sequences",
        help="Choose number of proteins to generate.",
        action="store",
        default=1
    )

    inv_fold_gen.add_argument(
        "--max_length",
        help="Max length of proteins generated, default is 500 AAs",
        default=500,
        type=int
    )

    inv_fold_gen.add_argument(
        "--top_p",
        help="ProstT5: If set to float < 1, only the smallest set of most probable tokens with probabilities that add "
             "up to top_p or higher are kept for generation. Default is 1",
        default=1
    )
    inv_fold_gen.add_argument(
        "--repetition_penalty",
        help="ProstT5: The parameter for repetition penalty. 1.0 means no penalty, the default is 1.2",
        default=1.2
    )
    inv_fold_gen.add_argument(
        "--dont_sample",
        help="ProstT5: By default, the model will sample to generate the protein. With this flag, you can enable "
             "greedy decoding, where the most probable tokens will be returned.",
        default=True,
        action="store_false"
    )
    # inv_fold_gen.add_argument("--mpnn_model", type=str, default="v_48_020",
    #                           help="ProteinMPNN Noise levels: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges "
    #                                "0.10A noise")
    inv_fold_gen.add_argument("--lig_mpnn_model", type=str, default="",
                              help="LigandMPNN: ProteinMPNN, Soluble, Global_Membrane, Local_Membrane, Side-Chain_Packing")
    inv_fold_gen.add_argument("--lig_mpnn_noise", type=str, default="010",
                              help="LigandMPNN Noise levels: 002, 005, 010, 020, 030; 010 = .10A noise. Note that 002 is only available for Soluble and Side-Chain_packing models")
    # inv_fold_gen.add_argument("--save_score", type=int, default=0,
    #                           help="ProteinMPNN: 0 for False, 1 for True; save score=-log_prob to npy files")
    # inv_fold_gen.add_argument("--save_probs", type=int, default=0,
    #                           help="ProteinMPNN: 0 for False, 1 for True; save MPNN predicted probabilities per "
    #                                "position")
    # inv_fold_gen.add_argument("--score_only", type=int, default=0,
    #                           help="ProteinMPNN: 0 for False, 1 for True; score input backbone-sequence pairs")
    # inv_fold_gen.add_argument("--path_to_fasta", type=str, default="",
    #                           help="ProteinMPNN: score provided input sequence in a fasta format; e.g. GGGGGG/PPPPS/WWW "
    #                                "for chains A, B, C sorted alphabetically and separated by /")
    # inv_fold_gen.add_argument("--conditional_probs_only", type=int, default=0,
    #                           help="ProteinMPNN: 0 for False, 1 for True; output conditional probabilities p(s_i given "
    #                                "the rest of the sequence and backbone)")
    # inv_fold_gen.add_argument("--conditional_probs_only_backbone", type=int, default=0,
    #                           help="ProteinMPNN: 0 for False, 1 for True; if true output conditional probabilities p(s_i "
    #                                "given backbone)")
    # inv_fold_gen.add_argument("--unconditional_probs_only", type=int, default=0,
    #                           help="ProteinMPNN: 0 for False, 1 for True; output unconditional probabilities p(s_i given "
    #                                "backbone) in one forward pass")
    # inv_fold_gen.add_argument("--backbone_noise", type=float, default=0.00,
    #                           help="ProteinMPNN: Standard deviation of Gaussian noise to add to backbone atoms")
    # inv_fold_gen.add_argument("--batch_size", type=int, default=1,
    #                           help="ProteinMPNN: Batch size; can set higher for titan, quadro GPUs, reduce this if "
    #                                "running out of GPU memory")
    # inv_fold_gen.add_argument("--pdb_path_chains", type=str, default="",
    #                           help="ProteinMPNN/LigandMPNN: Define which chains need to be designed for a single PDB ")
    # inv_fold_gen.add_argument("--chain_id_jsonl", type=str, default="",
    #                           help="ProteinMPNN: Path to a dictionary specifying which chains need to be designed and "
    #                                "which ones are fixed, if not specified all chains will be designed.")
    # inv_fold_gen.add_argument("--fixed_positions_jsonl", type=str, default="",
    #                           help="ProteinMPNN: Path to a dictionary with fixed positions")
    inv_fold_gen.add_argument("--omit_AAs", type=list, default="X",
                              help="LigandMPNN: Specify which amino acids should be omitted in the generated sequence, "
                                   "e.g. \"AC\" would omit alanine and cysteine.")
    # inv_fold_gen.add_argument("--bias_AA_jsonl", type=str, default="",
    #                           help="ProteinMPNN/LigandMPNN: Path to a dictionary which specifies AA composition bias if needed, "
    #                                "e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
    # inv_fold_gen.add_argument("--bias_by_res_jsonl", default="",
    #                           help="ProteinMPNN: Path to dictionary with per position bias.")
    # inv_fold_gen.add_argument("--omit_AA_jsonl", type=str, default="",
    #                           help="ProteinMPNN: Path to a dictionary which specifies which amino acids need to be omitted "
    #                                "from design at specific chain indices")
    # inv_fold_gen.add_argument("--pssm_jsonl", type=str, default="", help="ProteinMPNN: Path to a dictionary with pssm")
    # inv_fold_gen.add_argument("--pssm_multi", type=float, default=0.0,
    #                           help="ProteinMPNN: A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN "
    #                                "predictions")
    # inv_fold_gen.add_argument("--pssm_threshold", type=float, default=0.0,
    #                           help="ProteinMPNN: A value between -inf + inf to restrict per position AAs")
    # inv_fold_gen.add_argument("--pssm_log_odds_flag", type=int, default=0, help="ProteinMPNN: 0 for False, 1 for True")
    # inv_fold_gen.add_argument("--pssm_bias_flag", type=int, default=0, help="ProteinMPNN: 0 for False, 1 for True")
    # inv_fold_gen.add_argument("--tied_positions_jsonl", type=str, default="",
    #                           help="ProteinMPNN: Path to a dictionary with tied positions")

    inv_fold_gen.add_argument(
        "--fasta_seq_separation",
        type=str,
        default=":",
        help="LigandMPNN: Symbol to use between sequences from different chains",
    )
    inv_fold_gen.add_argument("--verbose", type=int, default=1, help="LigandMPNN: Print stuff")

    inv_fold_gen.add_argument(
        "--pdb_path_multi",
        type=str,
        default="",
        help="LigandMPNN: Path to json listing PDB paths. {'/path/to/pdb': ''} - only keys will be used.",
    )

    inv_fold_gen.add_argument(
        "--fixed_residues",
        type=str,
        default="",
        help="LigandMPNN: Provide fixed residues, A12 A13 A14 B2 B25",
    )
    inv_fold_gen.add_argument(
        "--fixed_residues_multi",
        type=str,
        default="",
        help="LigandMPNN: Path to json mapping of fixed residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}",
    )

    inv_fold_gen.add_argument(
        "--redesigned_residues",
        type=str,
        default="",
        help="LigandMPNN: Provide to be redesigned residues, everything else will be fixed, A12 A13 A14 B2 B25",
    )
    inv_fold_gen.add_argument(
        "--redesigned_residues_multi",
        type=str,
        default="",
        help="LigandMPNN: Path to json mapping of redesigned residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}",
    )

    inv_fold_gen.add_argument(
        "--bias_AA",
        type=str,
        default="",
        help="LigandMPNN: Bias generation of amino acids, e.g. 'A:-1.024,P:2.34,C:-12.34'",
    )
    inv_fold_gen.add_argument(
        "--bias_AA_per_residue",
        type=str,
        default="",
        help="LigandMPNN: Path to json mapping of bias {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}",
    )
    inv_fold_gen.add_argument(
        "--bias_AA_per_residue_multi",
        type=str,
        default="",
        help="LigandMPNN: Path to json mapping of bias {'pdb_path': {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}}",
    )

    inv_fold_gen.add_argument(
        "--omit_AA_per_residue",
        type=str,
        default="",
        help="LigandMPNN: Path to json mapping of bias {'A12': 'APQ', 'A13': 'QST'}",
    )
    inv_fold_gen.add_argument(
        "--omit_AA_per_residue_multi",
        type=str,
        default="",
        help="LigandMPNN: Path to json mapping of bias {'pdb_path': {'A12': 'QSPC', 'A13': 'AGE'}}",
    )

    inv_fold_gen.add_argument(
        "--symmetry_residues",
        type=str,
        default="",
        help="LigandMPNN: Add list of lists for which residues need to be symmetric, e.g. 'A12,A13,A14|C2,C3|A5,B6'",
    )
    inv_fold_gen.add_argument(
        "--symmetry_weights",
        type=str,
        default="",
        help="LigandMPNN: Add weights that match symmetry_residues, e.g. '1.01,1.0,1.0|-1.0,2.0|2.0,2.3'",
    )
    inv_fold_gen.add_argument(
        "--homo_oligomer",
        type=int,
        default=0,
        help="LigandMPNN: Setting this to 1 will automatically set --symmetry_residues and --symmetry_weights to do homooligomer design with equal weighting.",
    )

    inv_fold_gen.add_argument(
        "--zero_indexed",
        type=str,
        default=0,
        help="LigandMPNN: 1 - to start output PDB numbering with 0",
    )
    # inv_fold_gen.add_argument(
    #     "--seed",
    #     type=int,
    #     default=0,
    #     help="LigandMPNN: Set seed for torch, numpy, and python random.",
    # )
    inv_fold_gen.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="LigandMPNN: Number of sequence to generate per one pass.",
    )
    inv_fold_gen.add_argument(
        "--number_of_batches",
        type=int,
        default=1,
        help="LigandMPNN: Number of times to design sequence using a chosen batch size.",
    )
    # inv_fold_gen.add_argument(
    #     "--temperature",
    #     type=float,
    #     default=0.1,
    #     help="LigandMPNN: Temperature to sample sequences.",
    # )
    inv_fold_gen.add_argument(
        "--save_stats", type=int, default=0, help="LigandMPNN: Save output statistics"
    )

    inv_fold_gen.add_argument(
        "--ligand_mpnn_use_atom_context",
        type=int,
        default=1,
        help="LigandMPNN: 1 - use atom context, 0 - do not use atom context.",
    )
    inv_fold_gen.add_argument(
        "--ligand_mpnn_cutoff_for_score",
        type=float,
        default=8.0,
        help="LigandMPNN: Cutoff in angstroms between protein and context atoms to select residues for reporting score.",
    )
    inv_fold_gen.add_argument(
        "--ligand_mpnn_use_side_chain_context",
        type=int,
        default=0,
        help="LigandMPNN: Flag to use side chain atoms as ligand context for the fixed residues",
    )
    inv_fold_gen.add_argument(
        "--chains_to_design",
        type=str,
        default=None,
        help="LigandMPNN: Specify which chains to redesign, all others will be kept fixed.",
    )

    inv_fold_gen.add_argument(
        "--parse_these_chains_only",
        type=str,
        default="",
        help="LigandMPNN: Provide chains letters for parsing backbones, 'ABCF'",
    )

    inv_fold_gen.add_argument(
        "--transmembrane_buried",
        type=str,
        default="",
        help="LigandMPNN: Provide buried residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )
    inv_fold_gen.add_argument(
        "--transmembrane_interface",
        type=str,
        default="",
        help="LigandMPNN: Provide interface residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )

    inv_fold_gen.add_argument(
        "--global_transmembrane_label",
        type=int,
        default=0,
        help="LigandMPNN: Provide global label for global_label_membrane_mpnn model. 1 - transmembrane, 0 - soluble",
    )

    inv_fold_gen.add_argument(
        "--parse_atoms_with_zero_occupancy",
        type=int,
        default=0,
        help="LigandMPNN: To parse atoms with zero occupancy in the PDB input files. 0 - do not parse, 1 - parse atoms with zero occupancy",
    )

    inv_fold_gen.add_argument(
        "--pack_side_chains",
        type=int,
        default=0,
        help="LigandMPNN: 1 - to run side chain packer, 0 - do not run it",
    )

    inv_fold_gen.add_argument(
        "--number_of_packs_per_design",
        type=int,
        default=4,
        help="LigandMPNN: Number of independent side chain packing samples to return per design",
    )

    inv_fold_gen.add_argument(
        "--sc_num_denoising_steps",
        type=int,
        default=3,
        help="LigandMPNN: Number of denoising/recycling steps to make for side chain packing",
    )

    inv_fold_gen.add_argument(
        "--sc_num_samples",
        type=int,
        default=16,
        help="LigandMPNN: Number of samples to draw from a mixture distribution and then take a sample with the highest likelihood.",
    )

    inv_fold_gen.add_argument(
        "--repack_everything",
        type=int,
        default=0,
        help="LigandMPNN: 1 - repacks side chains of all residues including the fixed ones; 0 - keeps the side chains fixed for fixed residues",
    )

    inv_fold_gen.add_argument(
        "--force_hetatm",
        type=int,
        default=0,
        help="LigandMPNN: To force ligand atoms to be written as HETATM to PDB file after packing.",
    )

    inv_fold_gen.add_argument(
        "--packed_suffix",
        type=str,
        default="_packed",
        help="LigandMPNN: Suffix for packed PDB paths",
    )

    inv_fold_gen.add_argument(
        "--pack_with_ligand_context",
        type=int,
        default=1,
        help="LigandMPNN: 1-pack side chains using ligand context, 0 - do not use it.",
    )

def run(args):
    import os
    import shutil
    import subprocess
    import sys

    import pytorch_lightning as pl
    import torch
    from Bio import PDB
    from git import Repo
    from loguru import logger

    from trill.utils.esm_utils import ESM_IF1_Wrangle, ESM_IF1
    from trill.utils.lightning_models import ProstT5, Custom3DiDataset
    from trill.utils.inverse_folding.util import download_ligmpnn_weights
    from trill.utils.logging import setup_logger
    from .commands_common import cache_dir, get_logger

    ml_logger = get_logger(args)
    
    if args.model == "ESM-IF1":
        if args.query is None:
            raise Exception("A PDB or CIF file is needed for generating new proteins with ESM-IF1")
        data = ESM_IF1_Wrangle(args.query)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        sample_df, native_seq_df = ESM_IF1(dataloader, genIters=int(args.num_return_sequences), temp=float(args.temp),
                                           GPUs=int(args.GPUs))
        pdb_name = os.path.splitext(os.path.basename(args.query))[0]
        with open(os.path.join(args.outdir, f"{args.name}_ESM-IF1_gen.fasta"), "w+") as fasta:
            for ix, row in native_seq_df.iterrows():
                fasta.write(f">{pdb_name}_chain-{row[1]} \n")
                fasta.write(f"{row[0][0]}\n")
            for ix, row in sample_df.iterrows():
                fasta.write(f">{args.name}_ESM-IF1_chain-{row[1]} \n")
                fasta.write(f"{row[0]}\n")
    # elif args.model == "ProteinMPNN":
    #     if not os.path.exists((os.path.join(cache_dir, "ProteinMPNN/"))):
    #         logger.info("Cloning forked ProteinMPNN")
    #         os.makedirs(os.path.join(cache_dir, "ProteinMPNN/"))
    #         proteinmpnn = Repo.clone_from("https://github.com/martinez-zacharya/ProteinMPNN",
    #                                       (os.path.join(cache_dir, "ProteinMPNN", "")))
    #         mpnn_git_root = proteinmpnn.git.rev_parse("--show-toplevel")
    #         subprocess.run(("pip", "install", "-e", mpnn_git_root))
    #         sys.path.insert(0, (os.path.join(cache_dir, "ProteinMPNN", "")))
    #     else:
    #         sys.path.insert(0, (os.path.join(cache_dir, "ProteinMPNN", "")))
    #     from mpnnrun import run_mpnn
    #     logger.info("ProteinMPNN generation starting...")
    #     run_mpnn(args)

    elif args.model == "ProstT5":
        model = ProstT5(args)
        os.makedirs("foldseek_intermediates")
        create_db_cmd = ("foldseek", "createdb", os.path.abspath(args.query), "DB")
        subprocess.run(create_db_cmd, cwd="foldseek_intermediates")
        lndb_cmd = f"foldseek lndb DB_h DB_ss_h".split()
        subprocess.run(lndb_cmd, cwd="foldseek_intermediates")
        convert_cmd = ("foldseek", "convert2fasta", os.path.join("foldseek_intermediates", "DB_ss"),
                       os.path.join(args.outdir, args.name) + "_ss.3di")
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
            trainer = pl.Trainer(enable_checkpointing=False, logger=ml_logger, num_nodes=int(args.nodes))
        else:
            trainer = pl.Trainer(enable_checkpointing=False, devices=int(args.GPUs), accelerator="gpu", logger=ml_logger,
                                 num_nodes=int(args.nodes))
        with open(os.path.join(args.outdir, f"{args.name}_ProstT5_InvFold.fasta"), "w+") as fasta:
            for i in range(int(args.num_return_sequences)):

                out = trainer.predict(model, dataloader)
                for seq, chain_id in zip(out, chain_id_list):
                    fasta.write(f">{args.name}_ProstT5_InvFold_Chain-{chain_id}_{i} \n")
                    fasta.write(f"{seq}\n")
                fasta.flush()

    elif args.model == 'LigandMPNN':
        args.loguru = logger
        if not os.path.exists((os.path.join(cache_dir, "LigandMPNN/"))):
            logger.info("Cloning forked LigandMPNN")
            os.makedirs(os.path.join(cache_dir, "LigandMPNN/"))
            proteinmpnn = Repo.clone_from("https://github.com/martinez-zacharya/LigandMPNN",
                                          (os.path.join(cache_dir, "LigandMPNN", "")))
            mpnn_git_root = proteinmpnn.git.rev_parse("--show-toplevel")
            subprocess.run(("pip", "install", "-e", mpnn_git_root))
            sys.path.insert(0, (os.path.join(cache_dir, "LigandMPNN", "")))
        else:
            sys.path.insert(0, (os.path.join(cache_dir, "LigandMPNN", "")))
        
        logger.info("Looking for LigandMPNN model weights...")
        if not os.path.exists((os.path.join(cache_dir, "LigandMPNN_weights"))):
            logger.info("Downloading LigandMPNN model weights")
            download_ligmpnn_weights(cache_dir)
            logger.info(f"Finished downloading LigandMPNN model weights to {cache_dir}/LigandMPNN_weights")
        logger.info("Found LigandMPNN model weights!")
        from lig_mpnn_run import lig_mpnn
        logger.info("LigandMPNN generation starting...")
        args.number_of_batches = int(args.num_return_sequences)
        lig_mpnn(args, cache_dir)