def setup(subparsers):
    score = subparsers.add_parser("score", help="Use ESM-1v or ESM2 to score protein sequences or ProteinMPNN to score protein structures")

    score.add_argument(
        "scorer",
        help="Score protein sequences with ESM-1v, ESM2-650M or protein structures with ProteinMPNN",
        choices=("ESM2_150M", "ESM1v", "ESM2_650M", "ProteinMPNN")
    )
    score.add_argument(
        "query",
        help="Path to protein PDB file to score. Can also provide a .txt file with absolute paths to multiple PDBs",
        action="store",
    )
    score.add_argument("--mpnn_model", type=str, default="ProteinMPNN",
                              help="ProteinMPNN: ProteinMPNN, LigandMPNN, Local_Membrane, Global_Membrane and Soluble. Default is ProteinMPNN")
    score.add_argument("--lig_mpnn_noise", type=str, default="010",
                              help="ProteinMPNN Noise levels: 002, 005, 010, 020, 030; 010 = .10A noise. Note that 002 is only available for Soluble and Side-Chain_packing models")
    # score.add_argument(
    #     "--ligand",
    #     help="Ligand of interest to be simulated with input receptor",
    #     action="store",
    # )

    # score.add_argument(
    #     "--ligand_mpnn_use_side_chain_context",
    #     type=int,
    #     default=0,
    #     help="LigandMPNN: Flag to use side chain atoms as ligand context for the fixed residues",
    # )

    # score.add_argument(
    #     "--pdb_path_multi",
    #     type=str,
    #     default="",
    #     help="LigandMPNN: Path to json listing PDB paths. {'/path/to/pdb': ''} - only keys will be used.",
    # )

    score.add_argument(
        "--global_transmembrane_label",
        type=int,
        default=0,
        help="Provide global label for global_label_membrane_mpnn model. 1 - transmembrane, 0 - soluble",
    )
    score.add_argument(
        "--transmembrane_buried",
        type=str,
        default="",
        help="ProteinMPNN: Provide buried residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25 \
            If inputting a .txt file with absolute paths to .pdb's, make sure that all of the proteins have the same residue labels, \
            else you can provide a .csv file here where the first column is 'Label' and the second is 'Residues'.",
        nargs="*"
    )

    score.add_argument(
        "--transmembrane_interface",
        type=str,
        default="",
        help="ProteinMPNN: Provide interface residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25. \
            If inputting a .txt file with absolute paths to .pdb's, make sure that all of the proteins have the same residue labels, \
            else you can provide a .csv file here where the first column is 'Label' and the second is 'Residues'.",
        nargs="*"

    )

    score.add_argument(
        "--batch_transmembrane_csv",
        type=str,
        default="",
        help="ProteinMPNN: You can provide a .csv file to specify mutliple transmembrane buried/interface residues. \
            The first column should be called 'Label', the second 'transmembrane_buried' and the third 'transmembrane_interface'.",
    )

    score.add_argument(
        "--ligand_mpnn_cutoff_for_score",
        type=float,
        default=8.0,
        help="ProteinMPNN: Cutoff in angstroms between protein and context atoms to select residues for reporting score.",
    )

    # score.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=1,
    #     help="ProteinMPNN: Number of sequence to score per pass.",
    # )

    # score.add_argument(
    #     "--autoregressive_score",
    #     type=int,
    #     default=1,
    #     help="LigandMPNN: 1 - run autoregressive scoring function; p(AA_1|backbone); p(AA_2|backbone, AA_1) etc, 0 - False",
    # )

    # score.add_argument(
    #     "--single_aa_score",
    #     type=int,
    #     default=0,
    #     help="LigandMPNN: 1 - run single amino acid scoring function; p(AA_i|backbone, AA_{all except ith one}), 0 - False",
    # )

    # score.add_argument(
    #     "--use_sequence",
    #     type=int,
    #     default=1,
    #     help="LigandMPNN: 1 - get scores using amino acid sequence info; 0 - get scores using backbone info only",
    # )

from loguru import logger
from .commands_common import cache_dir, get_logger
import os
import subprocess
import sys
from git import Repo
import esm
from trill.utils.esm_utils import ESM_sampler, ESM1v, ESM2_650M, ESM2_150M
from trill.utils.inverse_folding.util import download_ligmpnn_weights
from tqdm import tqdm
import csv
 

def run(args):
    args.loguru = logger
    if args.scorer == 'ProteinMPNN':
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

        logger.info("Looking for ProteinMPNN model weights...")
        if not os.path.exists((os.path.join(cache_dir, "LigandMPNN_weights"))):
            logger.info("Downloading ProteinMPNN model weights")
            download_ligmpnn_weights(cache_dir)
            logger.info(f"Finished downloading ProteinMPNN model weights to {cache_dir}/LigandMPNN_weights")
        logger.info("Found ProteinMPNN model weights!")
        from score import ligmpnn_score
        logger.info(f"{args.mpnn_model} scoring starting...")
        args.verbose = 0
        args.single_aa_score = 0
        args.autoregressive_score = 1
        args.use_sequence = 1
        # args.lig_mpnn_noise = '010'
        args.ligand_mpnn_use_side_chain_context = 0
        args.batch_size = 1
        # args.ligand_mpnn_cutoff_for_score = 8.0
        # args.transmembrane_buried = ""
        # args.transmembrane_interface = ""
        ligmpnn_score(args, cache_dir)
    else:
        if int(args.GPUs) == 0:
            device = 'cpu'
            autocast_dev = 'cpu'
        else:
            device = 'gpu'
            autocast_dev = 'cuda'
        args.scorer = args.scorer + '()'
        sampler = ESM_sampler(eval(args.scorer), device)
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        labels = data[:][0]
        sequences = data[:][1]
        scores = sampler.log_likelihood_batch(sequences, False, device=autocast_dev)
        with open(f'{args.name}_{args.scorer[:-2]}_scores.csv', mode='w+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Label', f'{args.scorer[:-2]}_Score'])
            for label, score in zip(labels, scores):
                writer.writerow([label, score])