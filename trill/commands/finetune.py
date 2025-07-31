def setup(subparsers):
    finetune = subparsers.add_parser("finetune", help="Finetune protein language models")

    finetune.add_argument(
        "model",
        help="Choose the protein language model to finetune. Note that ESM2 is trained with the MLM objective, "
             "while ProtGPT2/ZymCTRL/ProGen2 are trained with the CLM objective. ZymCTRL must be finetuned \
                with --ctrl_tag specifying a Enzymatic Commission number.",
        action="store",
        choices=("esm2_t6_8M", "esm2_t12_35M", "esm2_t30_150M", "esm2_t33_650M", "esm2_t36_3B", "esm2_t48_15B",
                 "ProtGPT2", "progen2-small", "progen2-medium", "progen2-large", "progen2-oas",
                 "progen2-BFD90", "progen2-xlarge", "ZymCTRL")
    )

    finetune.add_argument(
        "query",
        help="Input fasta file. For ProGen2, you can provide a .csv file where the first column are absolute\
            paths to fasta files and the second column is the control tag related to that fasta file on the\
                same row.",
        action="store"
    )

    finetune.add_argument(
        "--epochs",
        help="Number of epochs for fine-tuning. Default is 10",
        action="store",
        default=10,
        dest="epochs",
    )

    finetune.add_argument(
        "--save_on_epoch",
        help="Saves a checkpoint on every successful epoch completed. WARNING, this could lead to rapid storage "
             "consumption",
        action="store_true",
        default=False,
    )

    finetune.add_argument(
        "--lr",
        help="Learning rate for optimizer. Default is 0.0001",
        action="store",
        default=0.0001,
        dest="lr",
    )

    finetune.add_argument(
        "--batch_size",
        help="Change batch-size number for fine-tuning. Default is 1",
        action="store",
        default=1,
        dest="batch_size",
    )

    finetune.add_argument(
        "--mask_fraction",
        help="ESM: Change fraction of amino acids masked for MLM training. Default is 0.15",
        action="store",
        default=0.15,
    )

    finetune.add_argument(
        "--pre_masked_fasta",
        help="ESM: Use this flag to specify that your input fasta will be pre-masked and does not need masking "
             "performed by TRILL. The sequences will still be randomly shuffled.",
        action="store",
        default=False,
    )

    finetune.add_argument(
        "--strategy",
        help="Change training strategy. Default is None. List of strategies can be found at \
            https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html. \
            For ProGen2 only, you can select either deepspeed_stage_1, deepspeed_stage_2 \
                deepspeed_stage_2_offload, deepspeed_stage_3 and deepspeed_stage_3_offload.",
        action="store",
        default="auto",
        dest="strategy",
    )

    finetune.add_argument(
        "--ctrl_tag",
        help="ZymCTRL: Choose an Enzymatic Commision (EC) control tag for finetuning ZymCTRL. Note that the tag must "
             "match all of the enzymes in the query fasta file. You can find all ECs here "
             "https://www.brenda-enzymes.org/index.php. You can also provide a control tag for ProGen2, which can be \
                any arbitrary string specifying a 'class' of proteins.",
        action="store"
    )

    finetune.add_argument(
        "--finetuned",
        help="Input path to your previously finetuned model to continue finetuning",
        action="store",
        default=False,
        dest="finetuned",
    )

    finetune.add_argument(
        "--eval",
        help="ProGen2: You can choose to withold a random proportion of the input data for evaluation to \
            check for overfitting. Input a float, like 0.25, which would hold-out 25%% of the data from \
                finetuning for evaluation after every epoch.",
        action="store",
        default=0,
        type=float
    )

    finetune.add_argument(
        "--grad_accum_steps",
        help="ProGen2: You can choose to change the number of steps to accumulate gradients\
            for before performing a backwards pass, will help with GPU vRAM usage.",
        action="store",
        default=1,
    )

    finetune.add_argument(
        "--scheduler",
        help="ProGen2: Choose the learning rate scheduler to use during training, default is constant.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt"
        ],
        action="store",
        default="constant",
    )

    finetune.add_argument(
        "--warmup_steps",
        help="ProGen2: Number of steps for a warmup ramping up to the set learning rate, default is 0.",
        action="store",
        default=0,
    )



def run(args):
    import os

    import esm
    import pytorch_lightning as pl
    import torch
    from Bio import SeqIO
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from tokenizers import Tokenizer
    from loguru import logger
    from trill.utils.esm_utils import premasked_FastaBatchedDataset
    from trill.utils.lightning_models import ESM, ProtGPT2, ZymCTRL
    from trill.utils.protgpt2_utils import ProtGPT2_wrangle
    from trill.utils.update_weights import weights_update
    from trill.utils.progen_utils import prepare_data, Protein_dataset, init_new_embeddings, load_data, create_deepspeed_config
    from .commands_common import get_logger, get_profiler

    ml_logger = get_logger(args)
    profiler = get_profiler(args)


    if 'progen2' in args.model:
        if not args.ctrl_tag:
            args.ctrl_tag = ''
        
        if args.save_on_epoch:
            args.save_on_epoch = 'epoch'
        else:
            args.save_on_epoch = 'no'

        ds_config = create_deepspeed_config(args)

        if int(args.GPUs) > 0:
            use_cpu = False
            fp16 = False
            bf16 = True
            
        else:
            use_cpu = True
            fp16 = False

        if not args.strategy:
            args.strategy = ''

        use_cpu = True if int(args.GPUs) == 0 else False
        train_path, eval_path = prepare_data(args, bidirectional=False, ctrl_tag=args.ctrl_tag)
        
        model = AutoModelForCausalLM.from_pretrained(f"hugohrban/{args.model}", trust_remote_code=True)
        tokenizer = Tokenizer.from_pretrained(f"hugohrban/{args.model}")
        tokenizer.enable_padding(
            direction="right", pad_id=0, pad_token="<|pad|>", length=1024
        )
        tokenizer.enable_truncation(max_length=1024)

        train_data, prefixes = load_data(train_path)
        test_data, prefixes = load_data(eval_path)
        tokenizer.add_tokens(prefixes)
        tokenizer.pad_token_id = 0
        train_data = Protein_dataset(train_data, tokenizer)
        test_data = Protein_dataset(test_data, tokenizer)

        init_new_embeddings(model, prefixes)

        if len(test_data) == 0:
            training_args = TrainingArguments(output_dir=args.outdir, per_device_train_batch_size=int(args.batch_size),
                            num_train_epochs=int(args.epochs), save_strategy=args.save_on_epoch, logging_strategy="epoch", log_level='debug',
                            learning_rate=float(args.lr), lr_scheduler_type=args.scheduler, save_only_model=True, use_cpu=use_cpu, seed=args.RNG_seed,
                            fp16=fp16, bf16=bf16, deepspeed=ds_config, gradient_accumulation_steps=int(args.grad_accum_steps))
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                )

            train_res = trainer.train()

        else:
            training_args = TrainingArguments(output_dir=args.outdir, eval_strategy="epoch", per_device_train_batch_size=int(args.batch_size),per_device_eval_batch_size=int(args.batch_size),
                                    num_train_epochs=int(args.epochs), save_strategy=args.save_on_epoch, logging_strategy="epoch", log_level='debug',
                                    learning_rate=float(args.lr), lr_scheduler_type=args.scheduler, save_only_model=True, use_cpu=use_cpu, seed=args.RNG_seed,
                                    fp16=fp16, fsdp=args.strategy, deepspeed=ds_config, gradient_accumulation_steps=int(args.grad_accum_steps))
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=test_data
            )

            train_res = trainer.train()

        trainer.model.save_pretrained(os.path.join(args.outdir, f'{args.name}_{args.model}_{args.epochs}.pt'), safe_serialization=False) 

    elif not args.pre_masked_fasta and 'progen2' not in args.model:
        data = esm.data.FastaBatchedDataset.from_file(args.query)
        len_data = len(data)
        if args.model == "ProtGPT2":
            model = ProtGPT2(args)
            if args.finetuned:
                model = ProtGPT2.load_from_checkpoint(args.finetuned, args=args, strict=False)
            tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
            seq_dict_df = ProtGPT2_wrangle(data, tokenizer)
            dataloader = torch.utils.data.DataLoader(seq_dict_df, shuffle=True, batch_size=int(args.batch_size),
                                                     num_workers=0)
            if args.save_on_epoch:
                checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
                if int(args.GPUs) == 0:
                    trainer = pl.Trainer(profiler=profiler, max_epochs=int(args.epochs), logger=ml_logger,
                                         num_nodes=int(args.nodes), callbacks=[checkpoint_callback],
                                         default_root_dir=f"{os.path.join(args.outdir, args.name)}_ckpt")
                else:
                    trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator="gpu",
                                         max_epochs=int(args.epochs), logger=ml_logger, num_nodes=int(args.nodes),
                                         precision=16, strategy=args.strategy, callbacks=[checkpoint_callback],
                                         default_root_dir=f"{os.path.join(args.outdir, args.name)}_ckpt")
            else:
                if int(args.GPUs) == 0:
                    trainer = pl.Trainer(profiler=profiler, max_epochs=int(args.epochs), logger=ml_logger,
                                         num_nodes=int(args.nodes), enable_checkpointing=False)
                else:
                    trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator="gpu",
                                         default_root_dir=f"{os.path.join(args.outdir, args.name)}_ckpt",
                                         max_epochs=int(args.epochs), logger=ml_logger, num_nodes=int(args.nodes),
                                         precision=16, strategy=args.strategy, enable_checkpointing=False)
            trainer.fit(model=model, train_dataloaders=dataloader)
            if "deepspeed" in str(args.strategy):
                save_path = os.path.join(
                    os.getcwd(),
                    f"{os.path.join(args.outdir, args.name)}_ckpt", "checkpoints",
                    f"epoch={int(args.epochs) - 1}-step={len_data * int(args.epochs)}.ckpt")
                output_path = os.path.join(args.outdir, f"{args.name}_ProtGPT2_{args.epochs}.pt")
                trainer.save_checkpoint(output_path)
                try:
                    convert_zero_checkpoint_to_fp32_state_dict(output_path, f"{output_path[0:-3]}_fp32.pt")
                except Exception as e:
                    logger.info(
                        f"Exception {e} has occurred on attempted save of your deepspeed trained model. If this has to "
                        f"do with CPU RAM, please try "
                        f"pytorch_lightning.utilities.deepspeedconvert_zero_checkpoint_to_fp32_state_dict("
                        f"your_checkpoint.ckpt, full_model.pt")
            elif str(args.strategy) in {"fsdp", "FSDP", "FullyShardedDataParallel"}:
                pass

            else:
                trainer.save_checkpoint(os.path.join(args.outdir, f"{args.name}_{args.model}_{args.epochs}.pt"))

        elif args.model == "ZymCTRL":
            model = ZymCTRL(args)
            seq_dict_df = ProtGPT2_wrangle(data, model.tokenizer)
            dataloader = torch.utils.data.DataLoader(seq_dict_df, shuffle=True, batch_size=int(args.batch_size),
                                                     num_workers=0)
            if args.save_on_epoch:
                checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
                if int(args.GPUs) == 0:
                    trainer = pl.Trainer(profiler=profiler, max_epochs=int(args.epochs), logger=ml_logger,
                                         num_nodes=int(args.nodes), callbacks=[checkpoint_callback],
                                         default_root_dir=f"{os.path.join(args.outdir, args.name)}_ckpt")
                else:
                    trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator="gpu",
                                         max_epochs=int(args.epochs), logger=ml_logger, num_nodes=int(args.nodes),
                                         precision=16, strategy=args.strategy, callbacks=[checkpoint_callback],
                                         default_root_dir=f"{os.path.join(args.outdir, args.name)}_ckpt")
            else:
                if int(args.GPUs) == 0:
                    trainer = pl.Trainer(profiler=profiler, max_epochs=int(args.epochs), logger=ml_logger,
                                         num_nodes=int(args.nodes), enable_checkpointing=False)
                else:
                    trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator="gpu",
                                         default_root_dir=f"{os.path.join(args.outdir, args.name)}_ckpt",
                                         max_epochs=int(args.epochs), logger=ml_logger, num_nodes=int(args.nodes),
                                         precision=16, strategy=args.strategy, enable_checkpointing=False)
            trainer.fit(model=model, train_dataloaders=dataloader)
            if "deepspeed" in str(args.strategy):
                save_path = os.path.join(
                    args.outdir,
                    f"{args.name}_ckpt", "checkpoints",
                    f"epoch={int(args.epochs) - 1}-step={len_data * int(args.epochs)}.ckpt")
                output_path = os.path.join(args.outdir, f"{args.name}_ZymCTRL_{args.epochs}.pt")
                trainer.save_checkpoint(output_path)
                try:
                    convert_zero_checkpoint_to_fp32_state_dict(output_path, f"{output_path[0:-3]}_fp32.pt")
                except Exception as e:
                    logger.error(
                        f"Exception {e} has occured on attempted save of your deepspeed trained model. If this has to"
                        f"do with CPU RAM, please try "
                        f"pytorch_lightning.utilities.deepspeedconvert_zero_checkpoint_to_fp32_state_dict("
                        f"your_checkpoint.ckpt, full_model.pt")
            elif str(args.strategy) in {"fsdp", "FSDP", "FullyShardedDataParallel"}:
                pass
            else:
                trainer.save_checkpoint(os.path.join(args.outdir, f"{args.name}_{args.model}_{args.epochs}.pt"))

        else:
            model_import_name = f"esm.pretrained.{args.model}_UR50D()"
            model = ESM(eval(model_import_name), float(args.lr), args)
            if args.finetuned:
                model = weights_update(model=ESM(eval(model_import_name), 0.0001, args),
                                       checkpoint=torch.load(args.finetuned))
            dataloader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=int(args.batch_size), num_workers=0,
                                                     collate_fn=model.alphabet.get_batch_converter())

            if args.strategy in {"deepspeed_stage_3", "deepspeed_stage_3_offload", "deepspeed_stage_2",
                                 "deepspeed_stage_2_offload"}:
                save_path = os.path.join(
                    args.outdir,
                    "checkpoints", f"epoch={int(args.epochs) - 1}-step={len_data * int(args.epochs)}.ckpt")
                output_path = os.path.join(args.outdir, f"{args.name}_{args.model}_{args.epochs}.pt")
                if args.save_on_epoch:
                    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
                    trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, callbacks=[checkpoint_callback],
                                         default_root_dir=f"{os.path.join(args.outdir, args.name)}_ckpt",
                                         accelerator="gpu", strategy=args.strategy, max_epochs=int(args.epochs),
                                         logger=ml_logger, num_nodes=int(args.nodes), precision=16)
                else:
                    trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler,
                                         default_root_dir=f"{os.path.join(args.outdir, args.name)}_ckpt",
                                         accelerator="gpu", strategy=args.strategy, max_epochs=int(args.epochs),
                                         logger=ml_logger, num_nodes=int(args.nodes), precision=16,
                                         enable_checkpointing=False)
                trainer.fit(model=model, train_dataloaders=dataloader)
                trainer.save_checkpoint(output_path)
                try:
                    convert_zero_checkpoint_to_fp32_state_dict(output_path, f"{output_path[0:-3]}_fp32.pt")
                except Exception as e:
                    logger.info(
                        f"Exception {e} has occurred on attempted save of your deepspeed trained model. If this has to"
                        f"do with CPU RAM, please try "
                        f"pytorch_lightning.utilities.deepspeedconvert_zero_checkpoint_to_fp32_state_dict("
                        f"your_checkpoint.ckpt, full_model.pt")
            else:
                if args.save_on_epoch:
                    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
                    if int(args.GPUs) == 0:
                        trainer = pl.Trainer(profiler=profiler, max_epochs=int(args.epochs),
                                             callbacks=[checkpoint_callback],
                                             default_root_dir=f"{os.path.join(args.outdir, args.name)}_ckpt",
                                             logger=ml_logger, num_nodes=int(args.nodes))
                    else:
                        trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator="gpu",
                                             callbacks=[checkpoint_callback],
                                             default_root_dir=f"{os.path.join(args.outdir, args.name)}_ckpt",
                                             strategy=args.strategy, max_epochs=int(args.epochs), logger=ml_logger,
                                             num_nodes=int(args.nodes), precision=16)
                else:
                    if int(args.GPUs) == 0:
                        trainer = pl.Trainer(profiler=profiler, accelerator="cpu", max_epochs=int(args.epochs),
                                             logger=ml_logger, num_nodes=int(args.nodes), enable_checkpointing=False)
                    else:
                        trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator="gpu",
                                             strategy=args.strategy, max_epochs=int(args.epochs), logger=ml_logger,
                                             num_nodes=int(args.nodes), precision=16, enable_checkpointing=False)
                trainer.fit(model=model, train_dataloaders=dataloader)
                trainer.save_checkpoint(os.path.join(args.outdir, f"{args.name}_{args.model}_{args.epochs}.pt"))

    else:
        actual_seqs = []
        actual_labels = []
        masked_seqs = []
        masked_labels = []
        with open(args.query) as infasta:
            for record in SeqIO.parse(infasta, "fasta"):
                actual_labels.append(record.id)
                actual_seqs.append(str(record.seq))

        with open(args.pre_masked_fasta) as infasta:
            for record in SeqIO.parse(infasta, "fasta"):
                masked_labels.append(record.id)
                masked_seqs.append(str(record.seq))

        if len(actual_seqs) != len(masked_seqs):
            raise Exception(
                f"The amount of sequences in {args.query}: {len(actual_seqs)} does not equal {args.pre_masked_fasta}: "
                f"{len(masked_seqs)} ")

        dict_masked = dict(zip(masked_labels, masked_seqs))
        dict_actual = dict(zip(actual_labels, actual_seqs))

        # Combine the sequences based on matching labels
        combined_list = [(label, dict_actual[label], dict_masked[label]) for label in actual_labels if
                         label in dict_masked]
        labels, actual_seqs, masked_seqs = zip(*combined_list)

        # Convert tuples to lists (optional, depending on further use)
        labels = list(labels)
        actual_seqs = list(actual_seqs)
        masked_seqs = list(masked_seqs)

        data = premasked_FastaBatchedDataset(labels, actual_seqs, masked_seqs)
        len_data = len(data)
        model_import_name = f"esm.pretrained.{args.model}_UR50D()"
        model = ESM(eval(model_import_name), float(args.lr), args)
        if args.finetuned:
            model = weights_update(model=ESM(eval(model_import_name), 0.0001, args),
                                   checkpoint=torch.load(args.finetuned))
        dataloader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=int(args.batch_size), num_workers=0,
                                                 collate_fn=model.alphabet.get_batch_converter(masked=True))
        if args.strategy in {"deepspeed_stage_3", "deepspeed_stage_3_offload", "deepspeed_stage_2",
                             "deepspeed_stage_2_offload"}:
            save_path = os.path.join(
                args.outdir,
                "checkpoints", f"epoch={int(args.epochs) - 1}-step={len_data * int(args.epochs)}.ckpt")
            output_path = os.path.join(args.outdir, f"{args.name}_{args.model}_{args.epochs}.pt")
            if args.save_on_epoch:
                checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
                trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, callbacks=[checkpoint_callback],
                                     default_root_dir=f"{os.path.join(args.outdir, args.name)}_ckpt", accelerator="gpu",
                                     strategy=args.strategy, max_epochs=int(args.epochs), logger=ml_logger,
                                     num_nodes=int(args.nodes), precision=16)
            else:
                trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler,
                                     default_root_dir=f"{os.path.join(args.outdir, args.name)}_ckpt", accelerator="gpu",
                                     strategy=args.strategy, max_epochs=int(args.epochs), logger=ml_logger,
                                     num_nodes=int(args.nodes), precision=16, enable_checkpointing=False)
            trainer.fit(model=model, train_dataloaders=dataloader)
            trainer.save_checkpoint(output_path)
            try:
                convert_zero_checkpoint_to_fp32_state_dict(output_path, f"{output_path[0:-3]}_fp32.pt")
            except Exception as e:
                logger.info(
                    f"Exception {e} has occurred on attempted save of your deepspeed trained model. If this has to do "
                    f"with CPU RAM, please try "
                    f"pytorch_lightning.utilities.deepspeedconvert_zero_checkpoint_to_fp32_state_dict("
                    f"your_checkpoint.ckpt, full_model.pt")
        else:
            if args.save_on_epoch:
                checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
                if int(args.GPUs) == 0:
                    trainer = pl.Trainer(profiler=profiler, max_epochs=int(args.epochs),
                                         callbacks=[checkpoint_callback],
                                         default_root_dir=f"{os.path.join(args.outdir, args.name)}_ckpt", logger=ml_logger,
                                         num_nodes=int(args.nodes))
                else:
                    trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator="gpu",
                                         callbacks=[checkpoint_callback],
                                         default_root_dir=f"{os.path.join(args.outdir, args.name)}_ckpt",
                                         strategy=args.strategy, max_epochs=int(args.epochs), logger=ml_logger,
                                         num_nodes=int(args.nodes), precision=16)
            else:
                if int(args.GPUs) == 0:
                    trainer = pl.Trainer(profiler=profiler, accelerator="cpu", max_epochs=int(args.epochs),
                                         logger=ml_logger, num_nodes=int(args.nodes), enable_checkpointing=False)
                else:
                    trainer = pl.Trainer(devices=int(args.GPUs), profiler=profiler, accelerator="gpu",
                                         strategy=args.strategy, max_epochs=int(args.epochs), logger=ml_logger,
                                         num_nodes=int(args.nodes), precision=16, enable_checkpointing=False)
            trainer.fit(model=model, train_dataloaders=dataloader)
            trainer.save_checkpoint(os.path.join(args.outdir, f"{args.name}_{args.model}_{args.epochs}.pt"))
