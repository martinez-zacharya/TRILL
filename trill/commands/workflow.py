def setup(subparsers):
    workflow = subparsers.add_parser("workflow", help="Perform workflow of interest")

    workflow.add_argument(
        "workflow",
        help="Choose workflow which chains together TRILL commands and utilities. ",
        action="store",
        choices=("foldtune",)
    )

    workflow.add_argument(
        "query",
        help="Input protein fasta file.",
        action="store"
    )

    workflow.add_argument(
        "--foldtune_rounds",
        help="Set the number of foldtuning iterations to perform. The default is 5.",
        action="store",
        default=5,
    )

    workflow.add_argument(
        "--num_to_generate_per_round",
        help="Set the number of proteins generated per round. Default is 1000.",
        action="store",
        default=1000,
    )

    workflow.add_argument(
        "--foldtune_generator",
        help="Language model used to generate/finetune each foldtuning round. Default protgpt2. "
             "zymctrl requires --ctrl_tag (an EC number). progen2 uses --progen2_model.",
        action="store",
        default="protgpt2",
        choices=("protgpt2", "zymctrl", "progen2"),
    )

    workflow.add_argument(
        "--ctrl_tag",
        help="ZymCTRL only: Enzyme Commission (EC) control tag (e.g. 4.2.1.1) that conditions "
             "ZymCTRL generation and finetuning every round. REQUIRED with --foldtune_generator "
             "zymctrl; the tag must match every enzyme in the input fasta.",
        action="store",
        default=None,
    )

    workflow.add_argument(
        "--progen2_model",
        help="ProGen2 size for --foldtune_generator progen2. progen2-large and progen2-BFD90 are "
             "excluded (broken HF config). "
             "Default progen2-small.",
        action="store",
        default="progen2-small",
        choices=("progen2-small", "progen2-medium", "progen2-oas", "progen2-xlarge"),
    )

    workflow.add_argument(
        "--finetune_strategy",
        help="Change the training strategy for finetuning. Use this is running out of vRAM!.",
        action="store",
        default=False,
        choices=("deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_2_offload", "deepspeed_stage_3", "deepspeed_stage_3_offload")
    )

    workflow.add_argument(
        "--finetune_batch_size",
        help="Change batch-size number for finetuning proteins. Default is 1, but with more GPU RAM, you can do more",
        action="store",
        default=1,
    )

    workflow.add_argument(
        "--embed_batch_size",
        help="Change batch-size number for embedding proteins. Default is 1, but with more GPU RAM, you can do more",
        action="store",
        default=1,
    )

    workflow.add_argument(
        "--lang_gen_batch_size",
        help="Change batch-size number for generating proteins. Default is 1, but with more GPU RAM, you can do more",
        action="store",
        default=1,
    )

    workflow.add_argument(
        "--fold_batch_size",
        help="Change batch-size number for folding proteins. Default is 1, but with more GPU RAM, you can do more",
        action="store",
        default=1,
    )

    workflow.add_argument(
        "--fast_folding",
        help="Use ProstT5 to speed up foldtuning by extracting 3di tokens from amino acid sequences directly instead of folding with ESMFold",
        action="store_true",
        default=False,
    )

    workflow.add_argument(
        "--selection_distance",
        help="How to rank structurally-valid generated sequences by embedding distance "
             "to the natural training set each round. 'min' (default) = nearest-neighbor "
             "L1 distance (the paper's 'semantic change'); 'max' = L1 distance to the "
             "farthest training point.",
        action="store",
        default="min",
        choices=("min", "max"),
    )

    workflow.add_argument(
        "--fold_tmscore_threshold",
        help="Keep a generated structure only if min(qtmscore, ttmscore) "
             "to its best-matching input is >= this global TM-score (default 0.5, the "
             "same-fold threshold).",
        action="store",
        type=float,
        default=0.5,
    )

    workflow.add_argument(
        "--fast_fold_evalue",
        help="Fast foldtuning (--fast_folding): keep a generated sequence if its best 3Di "
             "alignment E-value to any input is <= this value (default 1e-3). This is a "
             "structural-alphabet proxy, NOT a TM-score fold guarantee (ProstT5 3Di "
             "databases have no coordinates, so no TM-score is available).",
        action="store",
        type=float,
        default=1e-3,
    )

def run(args):
    import os

    import esm
    import pytorch_lightning as pl
    import torch
    import pandas as pd
    from transformers import EsmTokenizer, EsmForMaskedLM
    from trill.utils.foldtuning import foldtune
    from loguru import logger
    import requests
    from .commands_common import get_logger, cache_dir

    ml_logger = get_logger(args)
    args.cache_dir = cache_dir

    if args.workflow == 'foldtune':
        logger.info('Beginning Foldtuning')
        foldtune(args)  

        