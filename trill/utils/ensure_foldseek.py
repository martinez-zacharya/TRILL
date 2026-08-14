import os, sys, sysconfig, tarfile, urllib.request, tempfile, shutil, pathlib, subprocess, glob

BASE = "https://mmseqs.com/foldseek"
GPU_URL = f"{BASE}/foldseek-linux-gpu.tar.gz"
AVX2_URL = f"{BASE}/foldseek-linux-avx2.tar.gz"
SSE_URL = f"{BASE}/foldseek-linux-sse.tar.gz"
ARM64_URL = f"{BASE}/foldseek-linux-arm64.tar.gz"

MIN_GPU_CC = 7.5


def env_bin() -> pathlib.Path:
    return pathlib.Path(sysconfig.get_path("scripts"))

def gpu_sentinel_path() -> pathlib.Path:
    return env_bin() / "foldseek.gpu"

def gpu_build_installed() -> bool:
    return gpu_sentinel_path().exists()

def cpu_arch() -> str:
    return os.uname().machine  # "x86_64", "aarch64", etc.

def has_avx2() -> bool:
    try:
        with open("/proc/cpuinfo", "r") as f:
            return "avx2" in f.read().lower()
    except Exception:
        return False

def _env_force_gpu():
    """None if TRILL_FOLDSEEK_GPU is unset, else the user's explicit bool choice."""
    v = os.environ.get("TRILL_FOLDSEEK_GPU")
    if v is None:
        return None
    return v.strip().lower() not in ("", "0", "false", "no")

def nvidia_present() -> bool:
    return (bool(shutil.which("nvidia-smi"))
            or os.path.exists("/proc/driver/nvidia/version")
            or bool(glob.glob("/dev/nvidia[0-9]*")))

def gpu_arch_supported() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        caps = [float(x) for x in out.stdout.split() if x.strip()]
        return any(c >= MIN_GPU_CC for c in caps)
    except Exception:
        return False

def should_use_gpu_build() -> bool:
    if cpu_arch() not in ("x86_64", "amd64"):
        return False
    force = _env_force_gpu()
    if force is not None:
        return force
    return nvidia_present() and gpu_arch_supported()

def pick_url(gpu: bool = False) -> str:
    arch = cpu_arch()
    if arch in ("aarch64", "arm64"):
        return ARM64_URL
    if arch in ("x86_64", "amd64"):
        if gpu:
            return GPU_URL
        return AVX2_URL if has_avx2() else SSE_URL
    # Default to x86_64 SSE if unknown
    return SSE_URL


def _download_install(url: str, dest: pathlib.Path) -> None:
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


def _smoke_ok(dest: pathlib.Path) -> bool:
    try:
        subprocess.run([str(dest), "-h"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _set_gpu_sentinel(on: bool) -> None:
    sentinel = gpu_sentinel_path()
    if on:
        sentinel.write_text("gpu\n")
    elif sentinel.exists():
        sentinel.unlink()


def _cpu_fallback_urls(url: str) -> list:
    """CPU builds to try, in order, after a failed smoke test (deduped by caller)."""
    if cpu_arch() not in ("x86_64", "amd64"):
        return []
    if url == GPU_URL:
        return [AVX2_URL if has_avx2() else SSE_URL, SSE_URL]
    if url == AVX2_URL:
        return [SSE_URL]
    return []


def install_foldseek():
    dest = env_bin() / "foldseek"
    gpu = should_use_gpu_build()

    # Skip the download if a working foldseek of the desired kind (GPU vs CPU) is already
    # installed.
    if dest.exists() and gpu_build_installed() == gpu and _smoke_ok(dest):
        print(str(dest))
        return

    url = pick_url(gpu)
    _download_install(url, dest)

    if _smoke_ok(dest):
        _set_gpu_sentinel(gpu)
        print(str(dest))
        return

    _set_gpu_sentinel(False)
    tried = {url}
    for fb in _cpu_fallback_urls(url):
        if fb in tried:
            continue
        tried.add(fb)
        _download_install(fb, dest)
        if _smoke_ok(dest):
            break

    subprocess.run([str(dest), "-h"], check=False)
    print(str(dest))

if __name__ == "__main__":
    install_foldseek()
