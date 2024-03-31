import os

home_dir = os.path.expanduser("~")
cache_dir = os.path.join(home_dir, ".trill_cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

_logger = None


def get_logger(args):
    global _logger

    if not args.logger:
        return False

    if _logger is None:
        from pytorch_lightning.loggers import TensorBoardLogger
        _logger = TensorBoardLogger("logs")

    return _logger


_profiler = None


def get_profiler(args):
    global _profiler
    if not args.profiler:
        return None

    if _profiler is None:
        from pytorch_lightning.profilers import PyTorchProfiler

        _profiler = PyTorchProfiler(filename="test-logs")
    return _profiler
