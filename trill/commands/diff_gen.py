def setup(subparsers):
    diff_gen = subparsers.add_parser("diff_gen", help="Generate proteins using Denoising Diffusion models")
    
    diff_gen.add_argument(
        "model",
        help="Design proteins using diffusion models.",
        choices=("RFDiffusion", "Genie2")
    )
    diff_gen.add_argument(
        "--contigs",
        help="Generate proteins between these sizes in AAs. For example, --contig 100-200, "
             "will result in proteins in this range",
        action="store",
    )

    diff_gen.add_argument(
        "--RFDiffusion_Override",
        help="Change RFDiffusion model. For example, --RFDiffusion_Override ActiveSite will use ActiveSite_ckpt.pt "
             "for holding small motifs in place. ",
        action="store",
        default=False
    )

    diff_gen.add_argument(
        "--num_return_sequences",
        help="Number of sequences for RFDiffusion to generate. Default is 5",
        default=5,
        type=int,
    )

    diff_gen.add_argument(
        "--Inpaint",
        help="Residues to inpaint.",
        action="store",
        default=None
    )

    diff_gen.add_argument(
        "--query",
        help="Input pdb file for motif scaffolding, partial diffusion etc.",
        action="store",
    )

    # diff_gen.add_argument(
    #     "--sym",
    #     help="Use this flag to generate symmetrical oligomers.",
    #     action="store_true",
    #     default=False
    # )

    # diff_gen.add_argument(
    #     "--sym_type",
    #     help="Define residues that binder must interact with. For example, --hotspots A30,A33,A34 , where A is the "
    #          "chain and the numbers are the residue indices.",
    #     action="store",
    #     default=None
    # )

    diff_gen.add_argument(
        "--partial_T",
        help="Adjust partial diffusion sampling value.",
        action="store",
        default=None,
        type=int
    )

    diff_gen.add_argument(
        "--partial_diff_fix",
        help="Pass the residues that you want to keep fixed for your input pdb during partial diffusion. Note that "
             "the residues should be 0-indexed.",
        action="store",
        default=None
    )

    diff_gen.add_argument(
        "--hotspots",
        help="Define residues that binder must interact with. For example, --hotspots A30,A33,A34 , where A is the "
             "chain and the numbers are the residue indices.",
        action="store",
        default=None
    )

    diff_gen.add_argument(
        "--scale",
        help="Genie2: Adjust sampling noise scale between 0 and 1 inclusive. Higher number leads to more noise injected in the reverse denoising process. Default is 0.6 for unconditional sampling and 0.4 for motif scaffolding.",
        action="store",
        type=float
    )

    diff_gen.add_argument(
        "--motifs",
        help="Genie2: Select what motifs you want to scaffold in your input pdb file. You can select a motif by first entering what chain the motif is on, followed by a hyphen, the starting residue index and the end residue index. For example, like A-15-100, will attempt to scaffold the motif on chain A from residue 15 through 100. You can select multiple motifs by simply listing them sequentially, like 'A-15-100 B-100-150' will attempt to scaffold both motifs at the same time.",
        action="store",
        nargs="*"
    )


    # diff_gen.add_argument(
    #     "--RFDiffusion_yaml",
    #     help="Specify RFDiffusion params using a yaml file. Easiest option for complicated runs",
    #     action="store",
    #     default=None
    # )


def run(args):
    import os
    import subprocess
    import sys

    import requests
    from git import Repo
    from loguru import logger
    from trill.utils.genie2 import clone_and_install_genie2, download_genie2_weights, sample_genie2_unconditional, sample_genie2_conditional, add_remark_to_pdb
    from .commands_common import cache_dir
    from trill.utils.setup_dgl import setup_dgl, downgrade_pytorch, restore_pytorch

    # command = "conda install -c dglteam dgl-cuda11.7 -y -S -q".split(" ")
    # subprocess.run(command, check=True)

    if args.model == "RFDiffusion":
        # setup_dgl()
        # og_torch_ver = downgrade_pytorch()
        logger.info("Finding RFDiffusion weights... \n")
        if not os.path.exists((os.path.join(cache_dir, "RFDiffusion_weights"))):
            os.makedirs(os.path.join(cache_dir, "RFDiffusion_weights"))

            urls = (
                "http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt",
                "http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt",
                "http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt",
                "http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt",
                "http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt",
                "http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt",
                "http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt"
            )
            for url in urls:
                if not os.path.isfile(os.path.join(cache_dir, "RFDiffusion_weights", url.split('/')[-1])):
                    logger.info(f"Fetching {url}...")
                    response = requests.get(url)
                    with open(os.path.join(cache_dir, "RFDiffusion_weights", url.split('/')[-1]), "wb") as fp:
                        fp.write(response.content)

        if not os.path.exists(os.path.join(cache_dir, "RFDiffusion")):
            logger.info("Cloning forked RFDiffusion")
            os.makedirs(os.path.join(cache_dir, "RFDiffusion"))
            rfdiff = Repo.clone_from("https://github.com/martinez-zacharya/RFDiffusion",
                                    os.path.join(cache_dir, "RFDiffusion", ""))
            rfdiff_git_root = rfdiff.git.rev_parse("--show-toplevel")
            subprocess.run(("pip", "install", "-e", rfdiff_git_root))
            subprocess.run(("pip", "install", os.path.join(rfdiff_git_root, "env", "SE3Transformer")))
            sys.path.insert(0, os.path.join(cache_dir, "RFDiffusion"))

        else:
            sys.path.insert(0, os.path.join(cache_dir, "RFDiffusion"))
            git_repo = Repo(os.path.join(cache_dir, "RFDiffusion"), search_parent_directories=True)
            rfdiff_git_root = git_repo.git.rev_parse("--show-toplevel")
        from run_inference import run_rfdiff
        # if args.sym:
        #     run_rfdiff(os.path.join(rfdiff_git_root, "config", "inference", "symmetry.yaml"), args)
        # else:
        #     run_rfdiff(os.path.join(rfdiff_git_root, "config", "inference", "base.yaml"), args)
        run_rfdiff(os.path.join(rfdiff_git_root, "config", "inference", "base.yaml"), args)
        # restore_pytorch(og_torch_ver)
    
    elif args.model == 'Genie2':
        args.outdir = os.path.join(args.outdir, 'Genie2_output')
        os.makedirs(args.outdir, exist_ok=True)
        if not args.contigs:
            logger.error("You need to provide --contigs to perform generation with Genie2!")
            raise Exception("You need to provide --contigs to perform generation with Genie2!")
        clone_and_install_genie2(cache_dir)
        download_genie2_weights(cache_dir)
        if not args.query:
            logger.info('Performing unconditional generation with Genie2')
            sample_genie2_unconditional(args, cache_dir)
        else:
            if args.motifs:
                logger.info('Performing motif scaffolding with Genie2')
                fixed_pdb_path = add_remark_to_pdb(args.query, args.motifs, args.contigs, args.outdir)
                sample_genie2_conditional(args, cache_dir)
            else:
                logger.error("You need to provide --motifs to perform motif scaffolding with Genie2!")
                raise Exception("You need to provide --motifs to perform motif scaffolding with Genie2!")