import os, sys, sysconfig, tarfile, urllib.request, tempfile, shutil, pathlib, subprocess

def env_bin() -> pathlib.Path:
    return pathlib.Path(sysconfig.get_path("scripts"))  # .../.pixi/envs/default/bin

def cpu_arch() -> str:
    return os.uname().machine  # "x86_64", "aarch64", etc.

def has_avx2() -> bool:
    try:
        with open("/proc/cpuinfo", "r") as f:
            return "avx2" in f.read().lower()
    except Exception:
        return False

def pick_url() -> str:
    arch = cpu_arch()
    if arch in ("aarch64", "arm64"):
        return "https://mmseqs.com/foldseek/foldseek-linux-arm64.tar.gz"
    if arch in ("x86_64", "amd64"):
        return "https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz" if has_avx2() \
               else "https://mmseqs.com/foldseek/foldseek-linux-sse.tar.gz"
    # Default to x86_64 SSE if unknown
    return "https://mmseqs.com/foldseek/foldseek-linux-sse.tar.gz"

def install_foldseek():
    dest = env_bin() / "foldseek"
    url = pick_url()

    # download to temp
    with tempfile.TemporaryDirectory() as td:
        tgz = pathlib.Path(td, "foldseek.tgz")
        urllib.request.urlretrieve(url, tgz)
        with tarfile.open(tgz, "r:gz") as t:
            t.extractall(td)
        src = pathlib.Path(td, "foldseek", "bin", "foldseek")
        if not src.exists():
            raise RuntimeError(f"Unexpected archive layout for {url}")
        shutil.copy2(src, dest)
        os.chmod(dest, 0o755)

    # smoke test; if the AVX2 build fails, try SSE fallback on x86_64
    try:
        subprocess.run([str(dest), "-h"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        if cpu_arch() in ("x86_64", "amd64") and "avx2" in url:
            # try SSE
            with tempfile.TemporaryDirectory() as td:
                sse = "https://mmseqs.com/foldseek/foldseek-linux-sse.tar.gz"
                tgz = pathlib.Path(td, "foldseek.tgz")
                urllib.request.urlretrieve(sse, tgz)
                with tarfile.open(tgz, "r:gz") as t:
                    t.extractall(td)
                src = pathlib.Path(td, "foldseek", "bin", "foldseek")
                shutil.copy2(src, dest)
                os.chmod(dest, 0o755)
            # final test (don’t fail if it prints help to stderr)
            subprocess.run([str(dest), "-h"], check=False)

    # optional: export an override var for your code (harmless if ignored)
    print(str(dest))

if __name__ == "__main__":
    install_foldseek()
