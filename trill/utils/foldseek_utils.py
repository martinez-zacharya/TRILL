import os
import subprocess
# from trill.utils.externals import ensure_bin

# FOLDSEEK = ensure_bin("foldseek")


def _prostt5_model_present(path):
    """True if `path` holds a non-empty foldseek ProstT5 model (a file, or a directory
    that contains at least one non-empty file)."""
    if not path:
        return False
    if os.path.isfile(path):
        return os.path.getsize(path) > 0
    if os.path.isdir(path):
        return any(
            os.path.isfile(os.path.join(path, f)) and os.path.getsize(os.path.join(path, f)) > 0
            for f in os.listdir(path)
        )
    return False


def run_foldseek_databases(args):
    # Allow pointing at a preexisting foldseek ProstT5 model (e.g. a shared *.gguf) to skip
    override = os.environ.get("TRILL_PROSTT5_MODEL")
    if _prostt5_model_present(override):
        print(f"Using ProstT5 weights from TRILL_PROSTT5_MODEL: {override}")
        return override

    weights_path = os.path.join(args.cache_dir, "prostt5_weights")

    if _prostt5_model_present(weights_path):
        print(f"ProstT5 weights already exist at: {weights_path}")
        return weights_path

    command = ["foldseek", "databases", "ProstT5", weights_path, "tmp"]
    subprocess.run(command, check=True)
    return weights_path
