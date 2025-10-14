# Utils for Genie2

import os
import subprocess
import requests
import sys
import shutil
from loguru import logger

def download_genie2_weights(cache_dir):
    genie_weights_dir = os.path.join(cache_dir, 'Genie2_weights', 'test', 'checkpoints')
    os.makedirs(genie_weights_dir, exist_ok=True)
    
    base_url = "https://github.com/aqlaboratory/genie2/releases/download/v1.0.0"
    files = ["epoch.30.ckpt", "epoch.40.ckpt"]

    for file in files:
        epoch_num = file.split('.')[-2]  # Extract "30" or "40"
        new_filename = f"epoch={epoch_num}.ckpt"
        file_path = os.path.join(genie_weights_dir, new_filename)

        if not os.path.exists(file_path):
            logger.info(f"Downloading {file} as {new_filename}...")
            url = f"{base_url}/{file}"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Saved to {file_path}")
        else:
            logger.info(f"{new_filename} already exists at {file_path}, skipping download.")

def clone_and_install_genie2(cache_dir):
    repo_url = "https://github.com/aqlaboratory/genie2"
    target_path = os.path.join(cache_dir, "genie2")
    weights_dir = os.path.join(cache_dir, "Genie2_weights")
    config_src = os.path.join(target_path, "results", "base", "configuration")
    config_dst = os.path.join(weights_dir, "test")

    # Clone Genie2 if needed
    if os.path.exists(target_path):
        logger.info(f"Genie2 already exists at {target_path}. Skipping clone.")
    else:
        try:
            logger.info(f"Cloning Genie2 into {target_path}...")
            subprocess.run(["git", "clone", repo_url, target_path], check=True)
            logger.info("Clone complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cloning Genie2: {e}")
            raise

    # Install Genie2 in editable mode without dependencies
    try:
        logger.info(f"Installing Genie2 from {target_path}...")
        subprocess.run(
            ["pip", "install", "-e", ".", "--no-deps"],
            cwd=target_path,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info("Installation complete.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing Genie2: {e}")
        raise

    # Ensure Genie2_weights directory exists
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(config_dst, exist_ok=True)

    # Copy configuration file if not already present
    if os.path.exists(os.path.join(config_dst, 'configuration')):
        logger.info(f"Configuration already exists at {config_dst}. Skipping copy.")
    else:
        if os.path.exists(config_src):
            logger.info(f"Copying configuration from {config_src} to {config_dst}...")
            shutil.copy(config_src, config_dst)
            logger.info("Configuration copy complete.")
        else:
            logger.error(f"Warning: configuration file not found at {config_src}.")

    return target_path


def sample_genie2_unconditional(args, cache_dir):
    script_path = os.path.join(cache_dir, "genie2", "genie", "sample_unconditional.py")

    name = args.name
    epoch = 40
    scale = args.scale if args.scale else 0.6
    outdir = args.outdir
    num_samples = int(args.num_return_sequences)
    min_length = args.contigs.split('-')[0]
    max_length = args.contigs.split('-')[1]
    num_devices = int(args.GPUs)
    rootdir = os.path.abspath(os.path.join(cache_dir, 'Genie2_weights'))

    command = [
        "python3", script_path,
        "--name", name,
        "--epoch", str(epoch),
        "--scale", str(scale),
        "--outdir", outdir,
        "--num_samples", str(num_samples),
        "--min_length", min_length,
        "--max_length", max_length,
        "--num_devices", str(num_devices),
        "--rootdir", rootdir
    ]

    try:
        subprocess.run(command, check=True)
        logger.info(f"Sample generation complete. Output saved to {outdir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e}")

def sample_genie2_conditional(args, cache_dir):
    script_path = os.path.join(cache_dir, "genie2", "genie", "sample_scaffold.py")
    name = args.name
    epoch = 30
    scale = args.scale if args.scale else 0.4
    outdir = args.outdir
    num_samples = int(args.num_return_sequences)
    min_length = args.contigs.split('-')[0]
    max_length = args.contigs.split('-')[1]
    num_devices = int(args.GPUs)
    rootdir = os.path.abspath(os.path.join(cache_dir, 'Genie2_weights'))

    command = [
        "python3", script_path,
        "--name", name,
        "--epoch", str(epoch),
        "--scale", str(scale),
        "--outdir", outdir,
        "--num_samples", str(num_samples),
        "--num_devices", str(num_devices),
        "--rootdir", rootdir,
        "--datadir", args.outdir
    ]

    try:
        subprocess.run(command, check=True)
        logger.info(f"Sample generation complete. Output saved to {outdir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e}")

def add_remark_to_pdb(pdb_path, motifs, contigs, outdir):
    min_len, max_len = map(int, contigs.split('-'))
    remark_lines = []
    motif_ordered_residues = []  # List of (chain, residue_number) in input order

    # Parse motifs and build remarks and ordered residue list
    for segment in motifs:
        parts = segment.strip().split('-')
        if len(parts) == 3 and parts[0].isalpha():
            chain, start, end = parts[0], int(parts[1]), int(parts[2])
            line = f"{'REMARK 999 INPUT':<18}{chain:<1}{start:>4}{end:>4} {chain}".ljust(80)
            remark_lines.append(line)
            motif_ordered_residues.extend((chain, resi) for resi in range(start, end + 1))
        elif len(parts) == 2 and all(part.isdigit() for part in parts):
            start, end = map(int, parts)
            line = f"{'REMARK 999 INPUT':<18} {start:>4}{end:>4}".ljust(80)
            remark_lines.append(line)
            motif_ordered_residues.extend((' ', resi) for resi in range(start, end + 1))
        else:
            raise ValueError(f"Invalid segment format: {segment}")

    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    remark_lines.insert(0, f"REMARK 999 NAME {pdb_name}")
    remark_lines.append(f"REMARK 999 MINIMUM TOTAL LENGTH      {min_len}")
    remark_lines.append(f"REMARK 999 MAXIMUM TOTAL LENGTH      {max_len}")

    # Load original PDB lines and index them by (chain, residue)
    residue_atom_lines = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21].strip() or ' '
                resi = int(line[22:26])
                key = (chain, resi)
                residue_atom_lines.setdefault(key, []).append(line)
            elif line.startswith("END"):  # Skip END line; we’ll write it last
                continue

    # Collect PDB lines in the order specified by motifs
    ordered_pdb_lines = []
    for key in motif_ordered_residues:
        lines = residue_atom_lines.get(key, [])
        ordered_pdb_lines.extend(lines)

    # Write updated PDB file
    output_name = os.path.basename(pdb_path).split('.')[0] + '_with_remarks.pdb'
    output_path = os.path.join(outdir, output_name)
    with open(output_path, 'w') as f:
        for line in remark_lines:
            f.write(line + '\n')
        f.writelines(ordered_pdb_lines)
        f.write("END\n")

    logger.info(f"Updated PDB with REMARKS and reordered residues written to: {output_path}")
    return output_path