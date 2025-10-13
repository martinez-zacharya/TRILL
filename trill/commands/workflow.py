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

        