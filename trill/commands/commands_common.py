import os

home_dir = os.path.expanduser("~")
cache_dir = os.path.join(home_dir, ".trill_cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


def get_logger(args):
    if not args.logger:
        return False

    from pytorch_lightning.loggers import TensorBoardLogger

    return TensorBoardLogger("logs")


def get_profiler(args):
    if not args.profiler:
        return None

    from pytorch_lightning.profilers import PyTorchProfiler

    return PyTorchProfiler(filename="test-logs")
