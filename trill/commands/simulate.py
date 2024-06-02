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


def run(args):
    import os

    from trill.utils.dock_utils import fixer_of_pdbs
    from trill.utils.simulation_utils import relax_structure, run_simulation

    if args.just_relax:
        pdb_list = []
        if args.receptor.endswith(".txt"):
            with open(args.receptor, "r") as infile:
                for path in infile:
                    path = path.strip()
                    if not path:
                        continue
                    pdb_list.append(path)
        else:
            pdb_list.append(args.receptor)
        args.receptor = pdb_list
        fixed_pdb_files = fixer_of_pdbs(args)
        relax_structure(args, fixed_pdb_files)
    else:
        fixed_pdb_files = fixer_of_pdbs(args)

        args.output_traj_dcd = os.path.join(args.outdir, args.output_traj_dcd)

        # Run the simulation on the combined PDB file
        args.protein = fixed_pdb_files[0]
        run_simulation(args)
