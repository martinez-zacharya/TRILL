def setup(subparsers):
    score = subparsers.add_parser("score", help="Use COMPSS (ESM-1v and ProteinMPNN) to score proteins")

    score.add_argument(
        "query",
        help="Path to protein PDB file to score",
        action="store",
    )
    score.add_argument("--lig_mpnn_model", type=str, default="",
                              help="LigandMPNN: ProteinMPNN, Soluble, Global_Membrane, Local_Membrane, Side-Chain_Packing")
    score.add_argument("--lig_mpnn_noise", type=str, default="010",
                              help="LigandMPNN Noise levels: 002, 005, 010, 020, 030; 010 = .10A noise. Note that 002 is only available for Soluble and Side-Chain_packing models")
    score.add_argument(
        "--ligand",
        help="Ligand of interest to be simulated with input receptor",
        action="store",
    )

    score.add_argument(
        "--ligand_mpnn_use_side_chain_context",
        type=int,
        default=0,
        help="LigandMPNN: Flag to use side chain atoms as ligand context for the fixed residues",
    )

    score.add_argument(
        "--pdb_path_multi",
        type=str,
        default="",
        help="LigandMPNN: Path to json listing PDB paths. {'/path/to/pdb': ''} - only keys will be used.",
    )

    score.add_argument(
        "--transmembrane_buried",
        type=str,
        default="",
        help="LigandMPNN: Provide buried residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )
    score.add_argument(
        "--transmembrane_interface",
        type=str,
        default="",
        help="LigandMPNN: Provide interface residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )

    score.add_argument(
        "--ligand_mpnn_cutoff_for_score",
        type=float,
        default=8.0,
        help="LigandMPNN: Cutoff in angstroms between protein and context atoms to select residues for reporting score.",
    )

    score.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="LigandMPNN: Number of sequence to generate per one pass.",
    )

    score.add_argument(
        "--autoregressive_score",
        type=int,
        default=0,
        help="LigandMPNN: 1 - run autoregressive scoring function; p(AA_1|backbone); p(AA_2|backbone, AA_1) etc, 0 - False",
    )

    score.add_argument(
        "--single_aa_score",
        type=int,
        default=1,
        help="LigandMPNN: 1 - run single amino acid scoring function; p(AA_i|backbone, AA_{all except ith one}), 0 - False",
    )

    score.add_argument(
        "--use_sequence",
        type=int,
        default=1,
        help="LigandMPNN: 1 - get scores using amino acid sequence info; 0 - get scores using backbone info only",
    )
from loguru import logger
from .commands_common import cache_dir, get_logger
import os
import subprocess
import sys
from git import Repo
from trill.utils.inverse_folding.util import download_ligmpnn_weights
 

def run(args):

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
    from score import ligmpnn_score
    logger.info("LigandMPNN scoring starting...")
    args.verbose = 1
    ligmpnn_score(args, cache_dir)