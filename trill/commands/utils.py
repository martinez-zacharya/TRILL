def setup(subparsers):
    utils = subparsers.add_parser("utils", help="Misc utilities")

    utils.add_argument(
        "tool",
        help="prepare_class_key: Pepare a csv for use with the classify command. Takes a directory or text file with "
             "list of paths for fasta files. Each file will be a unique class, so if your directory contains 5 fasta "
             "files, there will be 5 classes in the output key csv. convert_to_safetensors: Convert PyTorch .pt files to safer safetensors format.",
        choices=("prepare_class_key", "fetch_embeddings", "convert_to_safetensors")
    )

    utils.add_argument(
        "--dir",
        help="Directory to be used for creating a class key csv for classification.",
        action="store",
    )

    utils.add_argument(
        "--fasta_paths_txt",
        help="Text file with absolute paths of fasta files to be used for creating the class key. Each unique path "
             "will be treated as a unique class, and all the sequences in that file will be in the same class.",
        action="store",
    )
    
    utils.add_argument(
        "--pt_file",
        help="PyTorch .pt file to convert to safetensors format.",
        action="store",
    )
    
    utils.add_argument(
        "--output_path",
        help="Output path for the converted safetensors file (optional).",
        action="store",
    )
    
    utils.add_argument(
        "--uniprotDB",
        help="UniProt embedding dataset to download.",
        choices=("UniProtKB",
                 "A.thaliana",
                 "C.elegans",
                 "E.coli",
                 "H.sapiens",
                 "M.musculus",
                 "R.norvegicus",
                 "SARS-CoV-2"),
        action="store",
    )
    utils.add_argument(
        "--rep",
        help="The representation to download.",
        choices=("per_AA", "avg"),
        action="store"
    )


def run(args):
    import os

    from trill.utils.classify_utils import generate_class_key_csv
    from trill.utils.fetch_embs import convert_embeddings_to_csv, download_embeddings
    from trill.utils.safe_load import convert_pt_to_safetensors

    if args.tool == "prepare_class_key":
        generate_class_key_csv(args)
    elif args.tool == "fetch_embeddings":
        h5_path = download_embeddings(args)
        h5_name = os.path.splitext(os.path.basename(h5_path))[0]
        convert_embeddings_to_csv(h5_path, os.path.join(args.outdir, f"{h5_name}.csv"))
    elif args.tool == "convert_to_safetensors":
        if not args.pt_file:
            raise ValueError("--pt_file is required for convert_to_safetensors")
        convert_pt_to_safetensors(args.pt_file, args.output_path)
