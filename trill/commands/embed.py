def setup(subparsers):
    embed = subparsers.add_parser("embed", help="Embed sequences of interest")

    embed.add_argument(
        "model",
        help="Choose language model to embed query sequences. Note for SaProt you need to protein structures as input \
            For RiNALMo, RNA-FM and mRNA-FM (must be multiples of 3 for mRNA-FM) the input is RNA while CaLM takes as input DNA sequences. MolT5/SMI-TED/SELFIES-TED all take smiles formatted small-molecules in fasta form as a .smiles input.",
        action="store",
        choices=("Ankh", "Ankh-Large", "CaLM", "esm2_t6_8M", "esm2_t12_35M", "esm2_t30_150M", "esm2_t33_650M", "esm2_t36_3B", "esm2_t48_15B",
                 "MMELON", "MolT5-Small", "MolT5-Base", "MolT5-Large", "ProtT5-XL", "ProstT5", "mRNA-FM", "RNA-FM", "SaProt", "SMI-TED", "SELFIES-TED")
    )
    #     choices=("Ankh", "Ankh-Large", "CaLM", "AMPLIFY_120M", "AMPLIFY_350M", "esm2_t6_8M", "esm2_t12_35M", "esm2_t30_150M", "esm2_t33_650M", "esm2_t36_3B", "esm2_t48_15B",
    #              "MolT5-Small", "MolT5-Base", "MolT5-Large", "ProtT5-XL", "ProstT5", "mRNA-FM", "RNA-FM", "SaProt")
    # )

    embed.add_argument(
        "query",
        help="Input fasta file. For SaProt only, you can provide a directory where every .pdb file will\
            be embedded or a .txt file where each line is an absolute path to a pdb file.",
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
        help="Add this flag to return the per amino acid / nucleic acid representations.",
        action="store_true",
        default=False,
    )
    embed.add_argument(
        "--avg",
        help="Add this flag to return the average, whole sequence representation.",
        action="store_true",
        default=False,
    )

    embed.add_argument(
        "--poolparti",
        help="ESM2/MolT5:Add this flag to return Pool PaRTI based embeddings.",
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
    from trill.utils.esm_utils import parse_and_save_all_predictions
    from trill.utils.lightning_models import ESM, CustomWriter, ProtT5, ProstT5, Ankh, SaProt, CaLM, RiNALMo, RNAFM, AMPLIFY
    from trill.utils.update_weights import weights_update
    from trill.utils.saprot_utils import get_struc_seq, preprocess_saprot, SaProt_Dataset, SaProt_Collator
    from trill.utils.molt5_utils import get_molT5_embed, prep_input_from_smiles_fasta
    from trill.utils.mmelon_utils import clone_and_install_mmelon, run_mmelon
    from trill.utils.materials_utils import clone_and_install_fm4m, run_mat_ted
    # from trill.utils.dplm import clone_and_install_dplm, downgrade_transformers, upgrade_transformer, get_transformer_version
    from loguru import logger
    import requests
    import subprocess
    import sys
    from .commands_common import get_logger, cache_dir
    from icecream import ic

    ml_logger = get_logger(args)

    if os.path.isdir(args.query):
        pass
    elif not args.query.endswith((".fasta", ".faa", ".fa", ".fna", ".smiles")):
        raise Exception(f"Input query file - {args.query} is not a valid file format.\
        File needs to be a fasta (.fa, .fasta, .faa, .fna, .smiles)")
    if not args.avg and not args.per_AA and not args.poolparti:
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
    
    elif 'MolT5' in args.model:
        avg_embs, aa_embs, headers, poolparti_df = get_molT5_embed(args)
        if args.avg:
            avg_embs.to_csv(os.path.join(args.outdir, f'{args.name}_{args.model}_AVG.csv'), index=False)
        if args.per_AA:
            data = {label: torch.tensor(embedding) for label, embedding in zip(headers, aa_embs)}
            torch.save(data, os.path.join(args.outdir, f'{args.name}_{args.model}_perAA.pt'))
        if args.poolparti:
            poolparti_df.to_csv(os.path.join(args.outdir, f'{args.name}_{args.model}_PoolParti.csv'), index=False)

    elif args.model == 'MMELON':
        clone_and_install_mmelon(cache_dir)
        headers_list, smiles_list = prep_input_from_smiles_fasta(args)
        embs = run_mmelon(args, smiles_list, headers_list)
        if args.avg:
            embs.to_csv(os.path.join(args.outdir, f'{args.name}_{args.model}_AVG.csv'), index=False)

    elif args.model == 'SMI-TED' or args.model == 'SELFIES-TED':
        clone_and_install_fm4m(cache_dir)
        headers_list, smiles_list = prep_input_from_smiles_fasta(args)
        embs = run_mat_ted(args, smiles_list, headers_list)
        if args.avg:
            embs.to_csv(os.path.join(args.outdir, f'{args.name}_{args.model}_AVG.csv'), index=False)
    # elif "AMPLIFY" in args.model:
    #     model = AMPLIFY(args)
    #     data = esm.data.FastaBatchedDataset.from_file(args.query)
    #     dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=int(args.batch_size), num_workers=0)
    #     pred_writer = CustomWriter(output_dir=args.outdir, write_interval="epoch")
    #     if int(args.GPUs) == 0:
    #         trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], logger=ml_logger,
    #                              num_nodes=int(args.nodes))
    #     else:
    #         trainer = pl.Trainer(enable_checkpointing=False, precision="16-mixed", devices=int(args.GPUs),
    #                              callbacks=[pred_writer], accelerator="gpu", logger=ml_logger, num_nodes=int(args.nodes))
    #     reps = trainer.predict(model, dataloader)
    #     cwd_files = os.listdir(args.outdir)
    #     pt_files = [file for file in cwd_files if "predictions_" in file]
    #     parse_and_save_all_predictions(args)

    #     for file in pt_files:
    #         os.remove(os.path.join(args.outdir, file))

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

    elif 'Ankh' in args.model:
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
    
    elif args.model == 'SaProt':
        model = SaProt(args)
        sa_seqs, labels = preprocess_saprot(args)
        data = SaProt_Dataset(sa_seqs, labels)
        collator = SaProt_Collator()
        dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=int(args.batch_size), num_workers=0, collate_fn=collator)
        pred_writer = CustomWriter(output_dir=args.outdir, write_interval="epoch")
        if int(args.GPUs) == 0:
            trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], logger=ml_logger,
                                 num_nodes=int(args.nodes))
        else:
            trainer = pl.Trainer(enable_checkpointing=False, precision=16, devices=int(args.GPUs),
                                 callbacks=[pred_writer], accelerator="gpu", logger=ml_logger, num_nodes=int(args.nodes))
        # if args.finetuned:
        #     model = weights_update(model=ESM(eval(model_import_name), 0.0001, args),
        #                            checkpoint=torch.load(args.finetuned))
        trainer.predict(model, dataloader)

        parse_and_save_all_predictions(args)

        cwd_files = os.listdir(args.outdir)
        pt_files = [file for file in cwd_files if "predictions_" in file]
        for file in pt_files:
            os.remove(os.path.join(args.outdir, file))


    elif args.model == 'CaLM':
        logger.info("Finding CaLM weights...")
        model_folder = os.path.join(cache_dir, "calm_weights")
        weights_file = os.path.join(model_folder, 'calm_weights.ckpt')
        if not os.path.exists(model_folder):
            logger.info('Downloading CaLM weights...')
            os.makedirs(model_folder, exist_ok=True)
            url = 'http://opig.stats.ox.ac.uk/data/downloads/calm_weights.pkl'
            with open(weights_file, 'wb') as handle:
                handle.write(requests.get(url).content)

        model = CaLM(args, weights_file)
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=int(args.batch_size), num_workers=0)
        pred_writer = CustomWriter(output_dir=args.outdir, write_interval="epoch")
        if int(args.GPUs) == 0:
            trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], logger=ml_logger,
                                 num_nodes=int(args.nodes))
        else:
            trainer = pl.Trainer(enable_checkpointing=False, precision=16, devices=int(args.GPUs),
                                 callbacks=[pred_writer], accelerator="gpu", logger=ml_logger, num_nodes=int(args.nodes))
        trainer.predict(model, dataloader)

        parse_and_save_all_predictions(args)

        cwd_files = os.listdir(args.outdir)
        pt_files = [file for file in cwd_files if "predictions_" in file]
        for file in pt_files:
            os.remove(os.path.join(args.outdir, file))

    elif args.model == 'RiNALMo':
        logger.info("Finding RiNALMo weights...")
        model_folder = os.path.join(cache_dir, "RiNALMo")
        weights_file = os.path.join(model_folder, 'rinalmo_giga_pretrained.pt')
        if not os.path.exists(model_folder):
            logger.info('Downloading RiNALMo weights...')
            os.makedirs(model_folder, exist_ok=True)
            url = 'https://zenodo.org/records/10725749/files/rinalmo_giga_pretrained.pt'
            with open(weights_file, 'wb') as handle:
                handle.write(requests.get(url).content)
        model = RiNALMo(args, weights_file)
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=int(args.batch_size), num_workers=0)
        pred_writer = CustomWriter(output_dir=args.outdir, write_interval="epoch")
        if int(args.GPUs) == 0:
            trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], logger=ml_logger,
                                 num_nodes=int(args.nodes))
        else:
            trainer = pl.Trainer(enable_checkpointing=False, precision=16, devices=int(args.GPUs),
                                 callbacks=[pred_writer], accelerator="gpu", logger=ml_logger, num_nodes=int(args.nodes))
        trainer.predict(model, dataloader)

        parse_and_save_all_predictions(args)

        cwd_files = os.listdir(args.outdir)
        pt_files = [file for file in cwd_files if "predictions_" in file]
        for file in pt_files:
            os.remove(os.path.join(args.outdir, file))

    elif args.model == 'RNA-FM' or args.model == 'mRNA-FM':
        logger.info(f"Finding {args.model} weights...")
        model = RNAFM(args)
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=int(args.batch_size), num_workers=0, collate_fn=model.alphabet.get_batch_converter())
        pred_writer = CustomWriter(output_dir=args.outdir, write_interval="epoch")
        if int(args.GPUs) == 0:
            trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], logger=ml_logger,
                                 num_nodes=int(args.nodes))
        else:
            trainer = pl.Trainer(enable_checkpointing=False, precision=16, devices=int(args.GPUs),
                                 callbacks=[pred_writer], accelerator="gpu", logger=ml_logger, num_nodes=int(args.nodes))
        trainer.predict(model, dataloader)

        parse_and_save_all_predictions(args)

        cwd_files = os.listdir(args.outdir)
        pt_files = [file for file in cwd_files if "predictions_" in file]
        for file in pt_files:
            os.remove(os.path.join(args.outdir, file))

    else:
        model_import_name = f"esm.pretrained.{args.model}_UR50D()"
        model = ESM(eval(model_import_name), 0.0001, args)
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=int(args.batch_size), num_workers=0,collate_fn=model.alphabet.get_batch_converter())
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
