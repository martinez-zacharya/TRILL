# Utils for using SCASA https://github.com/t-whalley/SCASA

import os
import subprocess
import pandas as pd
from icecream import ic
import re

def get_unique_chains(pdb_file):
    chains = set()
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21].strip()
                if chain_id:
                    chains.add(chain_id)
    return sorted(chains)

def calculate_sc_score(args):
    summary = []
    pdb_name = os.path.basename(args.query)

    command = ["SCASA", "sc", "--pdb", args.query, "--distance", str(args.interface_distance)]

    if args.chain_group_A:
        command += ["--complex_1", args.chain_group_A]
        print(f"Processing {pdb_name} with chain group {args.chain_group_A}...")
    else:
        chains = get_unique_chains(args.query)
        if len(chains) < 2:
            raise ValueError(f"Less than two unique chains found in {pdb_name}")
        if len(chains) > 2:
            raise ValueError(f"More than two unique chains found in {pdb_name}. Try passing --chain_group_A to score more than protein pairs")
        complex_1, complex_2 = chains[:2]
        command += ["--complex_1", complex_1, "--complex_2", complex_2]
        print(f"Processing {pdb_name} with chains {complex_1} and {complex_2}...")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    data = result.stdout.strip().splitlines()
    sc_value = None
    for line in data:
        match = re.search(r'\bSC\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line)
        if match:
            sc_value = float(match.group(1))
            break
    return sc_value