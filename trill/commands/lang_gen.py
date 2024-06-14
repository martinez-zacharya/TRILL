def setup(subparsers):
    lang_gen = subparsers.add_parser("lang_gen", help="Generate proteins using large language models")

    lang_gen.add_argument(
        "model",
        help="Choose desired language model",
        choices=["ESM2", "ProtGPT2", "ZymCTRL"]
    )
    lang_gen.add_argument(
        "--finetuned",
        help="Input path to your own finetuned model",
        action="store",
        default=False,
    )
    lang_gen.add_argument(
        "--esm2_arch",
        help="ESM2_Gibbs: Choose which ESM2 architecture your finetuned model is",
        action="store",
        default="esm2_t12_35M_UR50D",
    )
    lang_gen.add_argument(
        "--temp",
        help="Choose sampling temperature.",
        action="store",
        default="1",
    )

    lang_gen.add_argument(
        "--ctrl_tag",
        help="ZymCTRL: Choose an Enzymatic Commision (EC) control tag for conditional protein generation based on the "
             "tag. You can find all ECs here https://www.brenda-enzymes.org/index.php",
        action="store",
    )
    lang_gen.add_argument(
        "--batch_size",
        help="Change batch-size number to modulate how many proteins are generated at a time. Default is 1",
        action="store",
        default=1,
        dest="batch_size",
    )
    lang_gen.add_argument(
        "--seed_seq",
        help="Sequence to seed generation, the default is M.",
        default="M",
    )
    lang_gen.add_argument(
        "--max_length",
        help="Max length of proteins generated, default is 100",
        default=100,
        type=int
    )
    lang_gen.add_argument(
        "--do_sample",
        help="ProtGPT2/ZymCTRL: Whether or not to use sampling for generation; use greedy decoding otherwise",
        default=True,
        dest="do_sample",
    )
    lang_gen.add_argument(
        "--top_k",
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering",
        default=950,
        dest="top_k",
        type=int
    )
    lang_gen.add_argument(
        "--repetition_penalty",
        help="ProtGPT2/ZymCTRL: The parameter for repetition penalty, the default is 1.2. 1.0 means no penalty",
        default=1.2,
        dest="repetition_penalty",
    )
    lang_gen.add_argument(
        "--num_return_sequences",
        help="Number of sequences to generate. Default is 1",
        default=1,
        dest="num_return_sequences",
        type=int,
    )
    lang_gen.add_argument(
        "--random_fill",
        help="ESM2_Gibbs: Randomly select positions to fill each iteration for Gibbs sampling with ESM2. If not "
             "called then fill the positions in order",
        action="store_false",
        default=True,
    )
    lang_gen.add_argument(
        "--num_positions",
        help="ESM2_Gibbs: Generate new AAs for this many positions each iteration for Gibbs sampling with ESM2. If 0, "
             "then generate for all target positions each round.",
        action="store",
        default=0,
    )


def run(args):
    import os

    import torch
    import esm
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from loguru import logger
    from trill.utils.lightning_models import ProtGPT2, ESM_Gibbs, ZymCTRL
    from trill.utils.update_weights import weights_update

    if args.model == "ProtGPT2":
        model = ProtGPT2(args)
        if args.finetuned:
            model = model.load_from_checkpoint(args.finetuned, args=args, strict=False)
        tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
        generated_output = []
        total_sequences_needed = int(args.num_return_sequences)
        batch_size = int(args.batch_size)
        num_rounds = (total_sequences_needed + batch_size - 1) // batch_size

        with open(os.path.join(args.outdir, f"{args.name}_ProtGPT2.fasta"), "w+") as fasta:
            for round in tqdm(range(num_rounds)):
                num_sequences_this_round = batch_size if (round * batch_size + batch_size) <= total_sequences_needed \
                    else total_sequences_needed % batch_size

                generated_outputs = model.generate(
                    seed_seq=args.seed_seq,
                    max_length=int(args.max_length),
                    do_sample=args.do_sample,
                    top_k=int(args.top_k),
                    repetition_penalty=float(args.repetition_penalty),
                    num_return_sequences=num_sequences_this_round,
                    temperature=float(args.temp)
                )

                for i, generated_output in enumerate(generated_outputs):
                    fasta.write(f">{args.name}_ProtGPT2_{round * batch_size + i} \n")
                    fasta.write(f"{generated_output}\n")
                    fasta.flush()

    elif args.model == "ESM2":
        if int(args.GPUs) >= 1:
            logger.error(
                "*** Gibbs sampling on GPUs is currently down. For some reason, TRILL doesn't use generate different "
                "proteins regardless if a finetuned model is passed, but it works correctly on CPU... ***")
            raise RuntimeError
        model_import_name = f"esm.pretrained.{args.esm2_arch}()"
        with open(os.path.join(args.outdir, f"{args.name}_{args.esm2_arch}_Gibbs.fasta"), "w+") as fasta:
            model = ESM_Gibbs(eval(model_import_name), args)
            if int(args.GPUs) > 0:
                model.model = model.model.cuda()

            if args.finetuned:
                model = weights_update(model=ESM_Gibbs(eval(model_import_name), args),
                                       checkpoint=torch.load(args.finetuned))
                tuned_name = os.path.basename(args.finetuned)
            else:
                tuned_name = f"{args.esm2_arch}___"

            for i in range(args.num_return_sequences):
                out = model.generate(args.seed_seq, mask=True, n_samples=1, max_len=args.max_length,
                                     in_order=args.random_fill, num_positions=int(args.num_positions),
                                     temperature=float(args.temp))
                out = "".join(out)
                fasta.write(f">{args.name}_{tuned_name[0:-3]}_Gibbs_{i} \n")
                fasta.write(f"{out}\n")
                fasta.flush()

    elif args.model == "ZymCTRL":
        model = ZymCTRL(args)
        if args.finetuned:
            model = model.load_from_checkpoint(args.finetuned, args=args, strict=False)
        with open(os.path.join(args.outdir, f"{args.name}_ZymCTRL.fasta"), "w+") as fasta:
            for i in tqdm(range(int(args.num_return_sequences))):
                generated_output = model.generator(
                    str(args.ctrl_tag), device=torch.device("cpu" if int(args.GPUs) == 0 else "cuda"),
                    temperature=float(args.temp), max_length=int(args.max_length),
                    repetition_penalty=float(args.repetition_penalty), do_sample=args.do_sample,
                    top_k=int(args.top_k))

                fasta.write(f">{args.name}_{args.ctrl_tag}_ZymCTRL_{i}_PPL={generated_output[0][1]} \n")
                fasta.write(f"{generated_output[0][0]}\n")
                fasta.flush()
