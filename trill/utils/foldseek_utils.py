import os
import subprocess
# from trill.utils.externals import ensure_bin

# FOLDSEEK = ensure_bin("foldseek")

def run_foldseek_databases(args):
    weights_path = os.path.join(args.cache_dir, "prostt5_weights")

    if os.path.exists(weights_path):
        print(f"ProstT5 weights already exist at: {weights_path}")
        return weights_path

    command = ["foldseek", "databases", "ProstT5", weights_path, "tmp"]
    subprocess.run(command, check=True, shell=True)
    return weights_path

