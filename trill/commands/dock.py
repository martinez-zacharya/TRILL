def setup(subparsers):
    dock = subparsers.add_parser(
        "dock",
        help="Perform molecular docking with proteins and ligands. Note that you should relax your protein receptor "
             "with Simulate or another method before docking.")

    dock.add_argument(
        "algorithm",
        help="LightDock and GeoDock are only able to dock proteins-proteins currently. Vina, Smina, Gnina and DiffDock allow for docking small molecules to proteins.",
        choices=["DiffDock", "DiffDock-L", "Vina", "Smina", "Gnina", "LightDock", "GeoDock"]
    )

    dock.add_argument(
        "protein",
        help="Protein of interest to be docked with ligand",
        action="store"
    )

    dock.add_argument(
        "ligand",
        help="Ligand to dock protein with. Note that with Autodock Vina, you can dock multiple ligands at one time. "
             "Simply provide them one after another before any other optional TRILL arguments are added. Also, "
             "if a .txt file is provided with each line providing the absolute path to different ligands, TRILL will "
             "dock each ligand one at a time.",
        action="store",
        nargs="*"
    )

    # dock.add_argument(
    #     "--force_ligand",
    #     help="If you are not doing blind docking, TRILL will automatically assume your ligand is a small molecule if "
    #          "the MW is less than 800. To get around this, you can force TRILL to read the ligand as either type.",
    #     default=False,
    #     choices=["small", "protein"]
    # )

    dock.add_argument(
        "--save_visualisation",
        help="DiffDock: Save a pdb file with all of the steps of the reverse diffusion.",
        action="store_true",
        default=False
    )

    dock.add_argument(
        "--samples_per_complex",
        help="DiffDock: Number of samples to generate.",
        type=int,
        action="store",
        default=10
    )

    dock.add_argument(
        "--final_step_noise",
        help="DiffDock: Use noise in the final step of the reverse diffusion, the default is no noise in the final step.",
        action="store_true",
        default=False
    )

    dock.add_argument(
        "--inference_steps",
        help="DiffDock: Number of denoising steps",
        type=int,
        action="store",
        default=20
    )

    dock.add_argument(
        "--actual_steps",
        help="DiffDock: Number of denoising steps that are actually performed",
        type=int,
        action="store",
        default=0
    )

    # dock.add_argument(
    #     "--preComputed_Embs",
    #     help="DiffDock-L: Enter the path to your pre-computed embeddings. Make sure they are from esm2_t33_650M.",
    #     action="store",
    #     default=False
    # )


    dock.add_argument(
        "--min_radius",
        help="Smina/Vina + Fpocket: Minimum radius of alpha spheres in a pocket. Default is 3Å.",
        type=float,
        action="store",
        default=3.0
    )

    dock.add_argument(
        "--max_radius",
        help="Smina/Vina + Fpocket: Maximum radius of alpha spheres in a pocket. Default is 6Å.",
        type=float,
        action="store",
        default=6.0
    )

    dock.add_argument(
        "--min_alpha_spheres",
        help="Smina/Vina + Fpocket: Minimum number of alpha spheres a pocket must contain to be considered. Default "
             "is 35.",
        type=int,
        action="store",
        default=35
    )

    dock.add_argument(
        "--exhaustiveness",
        help="Smina/Vina: Change computational effort.",
        type=int,
        action="store",
        default=8
    )

    dock.add_argument(
        "--blind",
        help="Smina/Vina: Perform blind docking and skip binding pocket prediction with fpocket",
        action="store_true",
        default=False
    )
    dock.add_argument(
        "--anm",
        help="LightDock: If selected, backbone flexibility is modeled using Anisotropic Network Model (via ProDy)",
        action="store_true",
        default=False
    )

    dock.add_argument(
        "--swarms",
        help="LightDock: The number of swarms of the simulations, default is 25",
        action="store",
        type=int,
        default=25
    )

    dock.add_argument(
        "--glowworms",
        help="LightDock: The number of glowworms per swarm, default is 200",
        action="store",
        type=int,
        default=200
    )

    dock.add_argument(
        "--sim_steps",
        help="LightDock: The number of steps of the simulation. Default is 100",
        action="store",
        type=int,
        default=100
    )
    dock.add_argument(
        "--restraints",
        help="LightDock: If restraints_file is provided, residue restraints will be considered during the setup and "
             "the simulation",
        action="store_true",
        default=None
    )

    # dock.add_argument(
    #     "--membrane",
    #     help="LightDock: If selected, TRILL will insert your protein into a lipid bilary using Insane",
    #     action="store_true",
    #     default=None
    # )

    # dock.add_argument(
    #     "--lipid_upper",
    #     help="LightDock: Use to change the lipid composition for the upper leaflet (format: LIPID:ratio). Can be repeated like --lipid_upper POPC:1 DOPE:1",
    #     action="append",
    #     nargs="+",
    #     default=["POPC:1"]
    # )

    # dock.add_argument(
    #     "--lipid_lower",
    #     help="LightDock: Use to change the lipid composition for the lower leaflet (format: LIPID:ratio). Can be repeated",
    #     action="append",
    #     nargs="+",
    #     default=["POPC:1"]
    # )


def run(args):
    import os
    import subprocess
    import sys

    import esm
    import pkg_resources
    import pytorch_lightning as pl
    import requests
    import torch
    from esm.inverse_folding.util import load_coords
    from git import Repo
    import pandas as pd
    from icecream import ic
    from loguru import logger
    from trill.utils.dock_utils import perform_docking, write_docking_results_to_file, create_init_file
    from trill.utils.esm_utils import parse_and_save_all_predictions
    from trill.utils.lightning_models import ESM, CustomWriter
    from .commands_common import cache_dir, get_logger

    ml_logger = get_logger(args)

    if not args.algorithm == 'DiffDock-L':
        ligands = []
        if isinstance(args.ligand, list) and len(args.ligand) > 1:
            for lig in args.ligand:
                ligands.append(lig)
                args.multi_lig = True
        else:
            args.ligand = args.ligand[0]
            args.multi_lig = False
            if args.ligand.endswith(".txt"):
                with open(args.ligand, "r") as infile:
                    for path in infile:
                        path = path.strip()
                        if not path:
                            continue
                        ligands.append(path)
            else:
                ligands.append(args.ligand)

        protein_name = os.path.splitext(os.path.basename(args.protein))[0]

    if args.algorithm == "Smina" or args.algorithm == "Vina" or args.algorithm == "Gnina":
        args.cache_dir = cache_dir
        docking_results = perform_docking(args, ligands)
        write_docking_results_to_file(docking_results, args, protein_name, args.algorithm)
    elif args.algorithm == "LightDock":
        perform_docking(args, ligands)
        logger.info(f"LightDock run complete! Output files are in {args.outdir}")
    elif args.algorithm == "GeoDock":
        try:
            pkg_resources.get_distribution("geodock")
        except pkg_resources.DistributionNotFound:
            install_cmd = "pip install git+https://github.com/martinez-zacharya/GeoDock.git".split(" ")
            subprocess.run(install_cmd)
        from geodock.GeoDockRunner import EnMasseGeoDockRunner
        base_url = "https://raw.githubusercontent.com/martinez-zacharya/GeoDock/main/geodock/weights/dips_0.3.ckpt"
        weights_path = f"{cache_dir}/dips_0.3.ckpt"
        if not os.path.exists(weights_path):
            r = requests.get(base_url)
            with open(weights_path, "wb") as file:
                file.write(r.content)

        rec_coord, rec_seq = load_coords(args.protein, chain=None)
        rec_name = os.path.basename(args.protein).split(".")[0]

        lig_seqs = []
        lig_coords = []
        lig_names = []
        with open(f"tmp_master.fasta", "w+") as fasta:
            fasta.write(f">{rec_name}\n")
            fasta.write(f"{rec_seq}\n")
            for lig in ligands:
                lig_name = os.path.basename(lig).split(".")[0]
                coords, seq = load_coords(lig, chain=None)
                coords = torch.nan_to_num(torch.from_numpy(coords))
                lig_seqs.append(seq)
                lig_coords.append(coords)
                lig_names.append(lig_name)
                fasta.write(f">{lig_name}\n")
                fasta.write(f"{seq}\n")

        model_import_name = "esm.pretrained.esm2_t33_650M_UR50D()"
        args.per_AA = True
        args.avg = False
        model = ESM(eval(model_import_name), 0.0001, args)
        seq_data = esm.data.FastaBatchedDataset.from_file("tmp_master.fasta")
        loader = torch.utils.data.DataLoader(seq_data, shuffle=False, batch_size=1, num_workers=0,
                                             collate_fn=model.alphabet.get_batch_converter())
        pred_writer = CustomWriter(output_dir=args.outdir, write_interval="epoch")
        if int(args.GPUs) == 0:
            trainer = pl.Trainer(enable_checkpointing=False, callbacks=[pred_writer], logger=ml_logger,
                                 num_nodes=int(args.nodes))
        else:
            trainer = pl.Trainer(enable_checkpointing=False, precision=16, devices=int(args.GPUs),
                                 callbacks=[pred_writer], accelerator="gpu", logger=ml_logger, num_nodes=int(args.nodes))

        trainer.predict(model, loader)
        parse_and_save_all_predictions(args)
        master_embs = []
        emb_file = torch.load(os.path.join(args.outdir, "predictions_0.pt"), weights_only=False)
        for entry in emb_file[0]:
            emb = entry[0][0][0]
            master_embs.append(emb)

        rec_emb = master_embs.pop(0)
        for lig_name, lig_seq, lig_coord, lig_emb in zip(lig_names, lig_seqs, lig_coords, master_embs):
            em_geodock = EnMasseGeoDockRunner(args, ckpt_file=weights_path)
            pred = em_geodock.dock(
                rec_info=[rec_name, rec_seq, rec_coord, rec_emb],
                lig_info=[lig_name, lig_seq, lig_coord, lig_emb],
                out_name=args.name + "_" + rec_name + "_" + lig_name
            )
        os.remove(f"{args.outdir}/predictions_0.pt")

    elif args.algorithm == "DiffDock":
        if not os.path.exists(os.path.join(cache_dir, "DiffDock")):
            logger.info("Cloning forked DiffDock")
            os.makedirs(os.path.join(cache_dir, "DiffDock"))
            diffdock = Repo.clone_from("https://github.com/martinez-zacharya/DiffDock",
                                       os.path.join(cache_dir, "DiffDock"))
            diffdock_root = diffdock.git.rev_parse("--show-toplevel")
            subprocess.run(["pip", "install", "-e", diffdock_root])
            sys.path.insert(0, os.path.join(cache_dir, "DiffDock"))
        else:
            sys.path.insert(0, os.path.join(cache_dir, "DiffDock"))
            diffdock = Repo(os.path.join(cache_dir, "DiffDock"))
            diffdock_root = diffdock.git.rev_parse("--show-toplevel")
        from inference import run_diffdock
        run_diffdock(args, diffdock_root)

    elif args.algorithm == "DiffDock-L":
        tmp_csv_path = None
        if not os.path.exists(os.path.join(cache_dir, "DiffDock-L")):
            logger.info("Cloning DiffDock-L")
            os.makedirs(os.path.join(cache_dir, "DiffDock-L"))
            diffdock = Repo.clone_from("https://github.com/martinez-zacharya/DiffDock",
                                       os.path.join(cache_dir, "DiffDock-L"), branch='DiffDock-L')
            diffdock_root = diffdock.git.rev_parse("--show-toplevel")
            # subprocess.run(["pip", "install", "-e", diffdock_root])
            sys.path.insert(0, os.path.join(cache_dir, "DiffDock-L"))
        else:
            sys.path.insert(0, os.path.join(cache_dir, "DiffDock-L"))
            diffdock = Repo(os.path.join(cache_dir, "DiffDock-L"))
            diffdock_root = diffdock.git.rev_parse("--show-toplevel")
        directories = ['datasets', 'models', 'utils']
        for directory in directories:
            create_init_file(os.path.join(diffdock_root, directory))

        from inference import main

        if len(args.ligand) == 1 and args.protein.endswith('.pdb'):
            if args.final_step_noise:
                diffdock_l_cmd = f"python3 {cache_dir}/DiffDock-L/inference.py --config {cache_dir}/DiffDock-L/default_inference_args.yaml --protein_path {args.protein} --ligand {args.ligand[0]} --out_dir {args.outdir}".split()
            else:
                diffdock_l_cmd = f"python3 {cache_dir}/DiffDock-L/inference.py --config {cache_dir}/DiffDock-L/default_inference_args.yaml --protein_path {args.protein} --ligand {args.ligand[0]} --no_final_step_noise --out_dir {args.outdir}".split()

        elif args.protein.endswith('.csv'):
            pre_csv = pd.read_csv(args.protein)
            pre_csv['protein_sequence'] = ''
            tmp_csv_path = os.path.join(args.outdir, f'tmp_{args.name}_DiffDock-L_input.csv')
            pre_csv.to_csv(tmp_csv_path, index=None)
            # diffdock_l_cmd = f"python3 {cache_dir}/DiffDock-L/inference.py --config {cache_dir}/DiffDock-L/default_inference_args.yaml --protein_ligand_csv {tmp_csv_path} --out_dir {args.outdir}".split()
            diffdock_l_cmd = f"python3 {cache_dir}/DiffDock-L/inference.py --config {cache_dir}/DiffDock-L/default_inference_args.yaml --protein_ligand_csv {tmp_csv_path} --samples_per_complex {args.samples_per_complex} --actual_steps {args.actual_steps} --out_dir {args.outdir} --inference_steps {args.inference_steps} ".split()

        elif len(args.ligand) == 1 and args.ligand[0].endswith('.txt'):
            with open(args.ligand[0], 'r') as file:
                sdf_paths = file.readlines()
            data = []

            for sdf_path in sdf_paths:
                sdf_path = sdf_path.strip()
                if os.path.exists(sdf_path):
                    sdf_filename = os.path.basename(sdf_path)
                    sdf_name = os.path.splitext(sdf_filename)[0]
                    protein_name = os.path.splitext(os.path.basename(args.protein))[0]
                    complex_name = f"{protein_name}_{sdf_name}"
                    
                    row = {
                        'complex_name': complex_name,
                        'protein_path': args.protein,
                        'ligand_description': sdf_path,
                        'protein_sequence': ''
                    }
                    data.append(row)

            df = pd.DataFrame(data, columns=['complex_name', 'protein_path', 'ligand_description', 'protein_sequence'])
            tmp_csv_path = os.path.join(args.outdir, f'tmp_{args.name}_DiffDock-L_input.csv')
            df.to_csv(tmp_csv_path, index=False)
            if args.final_step_noise:
                diffdock_l_cmd = f"python3 {cache_dir}/DiffDock-L/inference.py --config {cache_dir}/DiffDock-L/default_inference_args.yaml --protein_ligand_csv {tmp_csv_path} --out_dir {args.outdir} --samples_per_complex {args.samples_per_complex} --inference_steps {args.inference_steps} --actual_steps {args.actual_steps} ".split()
            else:
                diffdock_l_cmd = f"python3 {cache_dir}/DiffDock-L/inference.py --config {cache_dir}/DiffDock-L/default_inference_args.yaml --protein_ligand_csv {tmp_csv_path} --out_dir {args.outdir} --samples_per_complex {args.samples_per_complex} --inference_steps {args.inference_steps} --no_final_step_noise --actual_steps {args.actual_steps} ".split()

        if args.save_visualisation:
            diffdock_l_cmd.append('--save_visualisation')
        diffdock_l_cmd.append('--inference_steps')
        diffdock_l_cmd.append(f'{args.inference_steps}')
        diffdock_l_cmd.append('--samples_per_complex')
        diffdock_l_cmd.append(f'{args.samples_per_complex}')
        ic(diffdock_l_cmd)
        subprocess.run(diffdock_l_cmd)
        if tmp_csv_path and os.path.exists(tmp_csv_path):
            os.remove(tmp_csv_path)

    
