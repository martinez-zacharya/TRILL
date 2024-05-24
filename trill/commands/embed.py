def setup(subparsers):
    embed = subparsers.add_parser("embed", help="Embed proteins of interest")

    embed.add_argument(
        "model",
        help="Choose protein language model to embed query proteins",
        action="store",
        choices=("esm2_t6_8M", "esm2_t12_35M", "esm2_t30_150M", "esm2_t33_650M", "esm2_t36_3B", "esm2_t48_15B",
                 "ProtT5-XL", "ProstT5", "Ankh", "Ankh-Large")
    )

    embed.add_argument(
        "query",
        help="Input protein fasta file",
        action="store"
    )

    embed.add_argument(
        "--batch_size",
        help="Change batch-size number for embedding proteins. Default is 1, but with more RAM, you can do more",
        action="store",
        default=1,
        dest="batch_size",
    )

    embed.add_argument(
        "--finetuned",
        help="Input path to your own finetuned ESM model",
        action="store",
        default=False,
        dest="finetuned",
    )

    embed.add_argument(
        "--per_AA",
        help="Add this flag to return the per amino acid representations.",
        action="store_true",
        default=False,
    )
    embed.add_argument(
        "--avg",
        help="Add this flag to return the average, whole sequence representation.",
        action="store_true",
        default=False,
    )


def run(args):
    import os

    import esm
    import pytorch_lightning as pl
    import torch

    from trill.utils.esm_utils import parse_and_save_all_predictions
    from trill.utils.lightning_models import ESM, CustomWriter, ProtT5, ProstT5, Ankh
    from trill.utils.update_weights import weights_update
    from loguru import logger
    from .commands_common import get_logger

    ml_logger = get_logger(args)

    if not args.query.endswith((".fasta", ".faa", ".fa")):
        raise Exception(f"Input query file - {args.query} is not a valid file format.\
        File needs to be a protein fasta (.fa, .fasta, .faa)")
    if not args.avg and not args.per_AA:
        logger.error("You need to select whether you want the average sequence embeddings or the per AA embeddings, or both!")
        raise RuntimeError
    if args.model == "ProtT5-XL":
        model = ProtT5(args)
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=int(args.batch_size), num_workers=0)
        pred_writer = CustomWriter(output_dir=args.outdir, write_interval="epoch")
        if int(args.GPUs) == 0:
            trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], logger=ml_logger,
                                 num_nodes=int(args.nodes))
        else:
            trainer = pl.Trainer(enable_checkpointing=False, precision=16, devices=int(args.GPUs),
                                 callbacks=[pred_writer], accelerator="gpu", logger=ml_logger, num_nodes=int(args.nodes))
        reps = trainer.predict(model, dataloader)
        cwd_files = os.listdir(args.outdir)
        pt_files = [file for file in cwd_files if "predictions_" in file]
        parse_and_save_all_predictions(args)

        for file in pt_files:
            os.remove(os.path.join(args.outdir, file))

    elif args.model == "ProstT5":
        model = ProstT5(args)
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=int(args.batch_size), num_workers=0)
        pred_writer = CustomWriter(output_dir=args.outdir, write_interval="epoch")
        if int(args.GPUs) == 0:
            trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], logger=ml_logger,
                                 num_nodes=int(args.nodes))
        else:
            trainer = pl.Trainer(enable_checkpointing=False, precision=16, devices=int(args.GPUs),
                                 callbacks=[pred_writer], accelerator="gpu", logger=ml_logger, num_nodes=int(args.nodes))

        reps = trainer.predict(model, dataloader)
        cwd_files = os.listdir(args.outdir)
        pt_files = [file for file in cwd_files if "predictions_" in file]
        parse_and_save_all_predictions(args)
        for file in pt_files:
            os.remove(os.path.join(args.outdir, file))

    elif args.model == "Ankh" or args.model == "Ankh-Large":
        model = Ankh(args)
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=int(args.batch_size),
                                                 num_workers=int(args.n_workers), persistent_workers=True)
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
        parse_and_save_all_predictions(args)
        for file in pt_files:
            os.remove(os.path.join(args.outdir, file))

    else:
        model_import_name = f"esm.pretrained.{args.model}_UR50D()"
        model = ESM(eval(model_import_name), 0.0001, args)
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=int(args.batch_size), num_workers=0,
                                                 collate_fn=model.alphabet.get_batch_converter())
        pred_writer = CustomWriter(output_dir=args.outdir, write_interval="epoch")
        if int(args.GPUs) == 0:
            trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], logger=ml_logger,
                                 num_nodes=int(args.nodes))
        else:
            trainer = pl.Trainer(enable_checkpointing=False, precision=16, devices=int(args.GPUs),
                                 callbacks=[pred_writer], accelerator="gpu", logger=ml_logger, num_nodes=int(args.nodes))
        if args.finetuned:
            model = weights_update(model=ESM(eval(model_import_name), 0.0001, args),
                                   checkpoint=torch.load(args.finetuned))
        trainer.predict(model, dataloader)

        parse_and_save_all_predictions(args)

        cwd_files = os.listdir(args.outdir)
        pt_files = [file for file in cwd_files if "predictions_" in file]
        for file in pt_files:
            os.remove(os.path.join(args.outdir, file))
