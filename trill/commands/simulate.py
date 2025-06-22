def setup(subparsers):
    simulate = subparsers.add_parser("simulate", help="Use OpenMM to perform molecular dynamics")

    simulate.add_argument(
        "receptor",
        help="Receptor of interest to be simulated. Must be either pdb file or a .txt file with the absolute path for "
             "each pdb, separated by a new-line.",
        action="store",
    )

    simulate.add_argument(
        "--ligand",
        help="Ligand of interest to be simulated with input receptor",
        action="store",
    )

    simulate.add_argument(
        "--constraints",
        help="Specifies which bonds and angles should be implemented with constraints. Allowed values are None, "
             "HBonds, AllBonds, or HAngles.",
        choices=("None", "HBonds", "AllBonds", "HAngles"),
        default="None",
        action="store",
    )

    simulate.add_argument(
        "--rigidWater",
        help="If true, water molecules will be fully rigid regardless of the value passed for the constraints argument.",
        default=None,
        action="store_true",
    )

    simulate.add_argument(
        "--forcefield",
        type=str,
        default="amber14-all.xml",
        help="Force field to use. Default is amber14-all.xml"
    )

    simulate.add_argument(
        "--solvent",
        type=str,
        choices=["implicit/hct.xml", "amber14/tip3p.xml", "amber14/tip3pfb.xml"],
        default="implicit/hct.xml",
        help="Solvent model to use. Options are 'implicit/hct.xml', 'amber14/tip3p.xml', or 'amber14/tip3pfb.xml'. The default is 'implicit/hct.xml'."
    )
    # simulate.add_argument(
    #     "--solvate",
    #     default=False,
    #     help="Add to solvate your simulation",
    #     action="store_true"
    # )

    simulate.add_argument(
        "--step_size",
        help="Step size in femtoseconds. Default is 2",
        type=float,
        default=2,
        action="store",
    )
    simulate.add_argument(
        "--num_steps",
        type=int,
        default=5000,
        help="Number of simulation steps"
    )

    simulate.add_argument(
        "--reporting_interval",
        type=int,
        default=1000,
        help="Reporting interval for simulation"
    )

    simulate.add_argument(
        "--output_traj_dcd",
        type=str,
        default="trajectory.dcd",
        help="Output trajectory DCD file"
    )

    simulate.add_argument(
        "--apply-harmonic-force",
        help="Whether to apply a harmonic force to pull the molecule.",
        type=bool,
        default=False,
        action="store",
    )

    simulate.add_argument(
        "--force-constant",
        help="Force constant for the harmonic force in kJ/mol/nm^2.",
        type=float,
        default=None,
        action="store",
    )

    simulate.add_argument(
        "--z0",
        help="The z-coordinate to pull towards in nm.",
        type=float,
        default=None,
        action="store",
    )

    simulate.add_argument(
        "--molecule-atom-indices",
        help="Comma-separated list of atom indices to which the harmonic force will be applied.",
        type=str,
        default="0,1,2",  # Replace with your default indices
        action="store",
    )

    simulate.add_argument(
        "--equilibration_steps",
        help="Steps you want to take for NVT and NPT equilibration. Each step is 0.002 picoseconds",
        type=int,
        default=300,
        action="store",
    )

    simulate.add_argument(
        "--periodic_box",
        help="Give, in nm, one of the dimensions to build the periodic boundary.",
        type=int,
        default=10,
        action="store",
    )

    simulate.add_argument(
        "--nonbonded_method",
        help="Specify the method for handling nonbonded interactions. Find more info in 3.6.5 of the OpenMM user guide.",
        type=str,
        choices=["NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "Ewald", "PME", "LJPME"],
        default="CutoffPeriodic",
        action="store",
    )
    # simulate.add_argument(
    #     "--martini_top",
    #     help="Specify the path to the MARTINI topology file you want to use.",
    #     type=str,
    #     default=False,
    #     action="store",
    # )
    simulate.add_argument(
        "--just_relax",
        help="Just relaxes the input structure(s) and outputs the fixed and relaxed structure(s). The forcefield that "
             "is used is amber14.",
        action="store_true",
        default=False,
    )

    simulate.add_argument(
        "--reporter_interval",
        help="Set interval to save PDB and energy snapshot. Note that the higher the number, the bigger the output "
             "files will be and the slower the simulation. Default is 1000",
        action="store",
        default=1000,
    )

    simulate.add_argument(
        "--rerun",
        help="Set to an integer to automatically re-try MD simulation in the case of 'Energy is NaN' error from OpenMM",
        action="store",
        type=int,
        default=1,
    )

def run(args):
    import os
    import pandas as pd
    from trill.utils.dock_utils import fixer_of_pdbs, run_vina_split, convert_pdbqt_to_mol2
    from trill.utils.simulation_utils import relax_structure, run_simulation
    from icecream import ic
    from loguru import logger

    if args.just_relax:
        pdb_list = []
        if args.receptor.endswith(".txt"):
            with open(args.receptor, "r") as infile:
                for path in infile:
                    path = path.strip()
                    if path:
                        pdb_list.append(path)
        else:
            pdb_list.append(args.receptor)

        args.receptor = pdb_list
        fixed_pdb_files = fixer_of_pdbs(args)
        relax_structure(args, fixed_pdb_files)
        return

    args.ogname = args.name

    fixed_pdb_files = fixer_of_pdbs(args)
    args.output_traj_dcd = os.path.join(args.outdir, args.output_traj_dcd)
    args.protein = fixed_pdb_files[0]
    max_retries = int(getattr(args, 'rerun', 0))

    master_df = pd.DataFrame()
    master_rmsd_df = pd.DataFrame()
    master_rmsf_df = pd.DataFrame()
    master_prolif_df = pd.DataFrame()

    def try_run_simulation(args):
        attempt = 0
        while attempt <= max_retries:
            # try:
            gbsa_df, rms_output, prolif_df = run_simulation(args)

            # RMSD
            rmsd_df = pd.DataFrame(rms_output['rmsd_complex'], columns=[f'RMSD_pose-{ix+1}'])
            rmsd_df['Frame'] = rmsd_df.index + 1
            rmsd_df = rmsd_df[['Frame', f'RMSD_pose-{ix+1}']]
            rmsd_df['Simulation_Path'] = args.sim_output_path
            rmsd_df.rename(columns={'Simulation_Path': f'Simulation_Path_pose-{ix+1}'}, inplace=True)

            # RMSF
            rmsf_df = pd.DataFrame(rms_output['rmsf_complex'], columns=[f'RMSF_pose-{ix+1}'])
            rmsf_df['Residue'] = rmsf_df.index + 1
            rmsf_df = rmsf_df[['Residue', f'RMSF_pose-{ix+1}']]
            rmsf_df['Simulation_Path'] = args.sim_output_path
            rmsf_df.rename(columns={'Simulation_Path': f'Simulation_Path_pose-{ix+1}'}, inplace=True)

            if gbsa_df is not None:
                gbsa_df["ligand"] = args.ligand
                gbsa_df.to_csv(os.path.join(args.outdir, f'{args.name}_MMGBSA_output.csv'), index=False)

            if prolif_df is not None:
                prolif_df["ligand"] = args.ligand
                prolif_df.to_csv(os.path.join(args.outdir, f'{args.name}_prolif_output.csv'), index=False)

            return gbsa_df, rmsd_df, rmsf_df, prolif_df

            # except Exception as e:
            #     logger.warning(f"[Attempt {attempt + 1}] Simulation failed with error:\n{e}")
            #     attempt += 1
            #     if attempt > max_retries:
            #         logger.error("Maximum retry attempts reached. Aborting.")
            #         return None, None, None, None

    if args.ligand.endswith('.pdbqt'):
        args.multi = True
        patt, max_model = run_vina_split(args.ligand)
        output_ligs = convert_pdbqt_to_mol2(patt)
        for ix, lig in enumerate(output_ligs):
            args.ligand = lig
            if ix == 0:
                args.name = args.name + f'_pose-{ix+1}'
            else:
                args.name = args.name.replace(f'_pose-{ix}', f'_pose-{ix+1}')

            logger.info(f"Preparing simulation for ligand pose {ix+1}: {lig}")
            gbsa_df, rmsd_df, rmsf_df, prolif_df = try_run_simulation(args)

            if gbsa_df is not None:
                master_df = pd.concat([master_df, gbsa_df], ignore_index=True)
            if rmsd_df is not None:
                if master_rmsd_df.empty:
                    master_rmsd_df = rmsd_df
                else:
                    master_rmsd_df = pd.concat([master_rmsd_df, rmsd_df.set_index('Frame')], axis=1).reset_index()
            if rmsf_df is not None:
                if master_rmsf_df.empty:
                    master_rmsf_df = rmsf_df
                else:
                    master_rmsf_df = pd.concat([master_rmsf_df, rmsf_df.set_index('Residue')], axis=1).reset_index()
            if prolif_df is not None:
                master_prolif_df = pd.concat([master_prolif_df, prolif_df], ignore_index=True)

            if gbsa_df is None:
                logger.warning(f"Skipping to next ligand after failed attempts for pose {ix+1}\n")

    else:
        args.multi = True
        ix = 0  # Single ligand
        gbsa_df, rmsd_df, rmsf_df, prolif_df = try_run_simulation(args)

        if gbsa_df is not None:
            gbsa_df["ligand"] = args.ligand
            master_df = gbsa_df
        if rmsd_df is not None:
            master_rmsd_df = rmsd_df
        if rmsf_df is not None:
            master_rmsf_df = rmsf_df
        if prolif_df is not None:
            prolif_df["ligand"] = args.ligand
            prolif_df.to_csv(os.path.join(args.outdir, f'{args.name}_prolif_output.csv'), index=False)
            master_prolif_df = prolif_df

    if not master_df.empty:
        logger.info("All successful GBSA results have been collected.")
        master_df.to_csv(os.path.join(args.outdir, f"{args.ogname}_master_MMGBSA.csv"), index=False)
        if not master_rmsd_df.empty:
            master_rmsd_df.to_csv(os.path.join(args.outdir, f"{args.ogname}_master_RMSD.csv"), index=False)
        if not master_rmsf_df.empty:
            master_rmsf_df.to_csv(os.path.join(args.outdir, f"{args.ogname}_master_RMSF.csv"), index=False)
        if not master_prolif_df.empty:
            master_prolif_df.to_csv(os.path.join(args.outdir, f"{args.ogname}_master_ProLIF_output.csv"), index=False)
        return master_df, master_rmsd_df, master_rmsf_df, master_prolif_df
    else:
        logger.warning("No successful simulations were completed.")
        return None, None, None, None

