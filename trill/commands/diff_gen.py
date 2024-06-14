def setup(subparsers):
    diff_gen = subparsers.add_parser("diff_gen", help="Generate proteins using RFDiffusion")

    diff_gen.add_argument(
        "--contigs",
        help="Generate proteins between these sizes in AAs for RFDiffusion. For example, --contig 100-200, "
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
    from .commands_common import cache_dir

    # command = "conda install -c dglteam dgl-cuda11.7 -y -S -q".split(" ")
    # subprocess.run(command, check=True)
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
