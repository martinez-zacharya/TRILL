import argparse
import importlib
import os
import sys
import time

import pytorch_lightning as pl
import torch
from pyfiglet import Figlet
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from transformers import set_seed

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):

    # torch.set_float32_matmul_precision('medium')
    start = time.time()
    f = Figlet(font="graffiti")
    print(f.renderText("TRILL"))
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "name",
        help = "Name of run",
        action = "store"
        )

    
    parser.add_argument(
        "GPUs",
        help="Input total number of GPUs per node",
        action="store",
        default = 1
)

    parser.add_argument(
        "--nodes",
        help="Input total number of nodes. Default is 1",
        action="store",
        default = 1
)
    

    parser.add_argument(
        "--logger",
        help="Enable Tensorboard logger. Default is None",
        action="store",
        default = False,
        dest="logger",
)

    parser.add_argument(
        "--profiler",
        help="Utilize PyTorchProfiler",
        action="store_true",
        default=False,
        dest="profiler",
)
    parser.add_argument(
        "--RNG_seed",
        help="Input RNG seed. Default is 123",
        action="store",
        default = 123
)
    parser.add_argument(
        "--outdir",
        help="Input full path to directory where you want the output from TRILL",
        action="store",
        default = '.'
)

    parser.add_argument(
        "--n_workers",
        help="Change number of CPU cores/'workers' TRILL uses",
        action="store",
        default = 1
)


##############################################################################################################

    subparsers = parser.add_subparsers(dest='command')

    commands = {}
    for command in [
        "embed",
        "finetune",
        "inv_fold_gen",
        "lang_gen",
        "diff_gen",
        "classify",
        "fold",
        "visualize",
        "simulate",
        "dock",
        "utils",
    ]:
        commands[command] = importlib.import_module(f"trill.commands.{command}")
        commands[command].setup(subparsers)

    args = parser.parse_args()

    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".trill_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    pl.seed_everything(int(args.RNG_seed))
    set_seed(int(args.RNG_seed))

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    
    
    torch.backends.cuda.matmul.allow_tf32 = True
    if int(args.GPUs) == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if int(args.nodes) <= 0:
            raise Exception(f'There needs to be at least one cpu node to use TRILL')
    #if args.tune == True:
        #data = esm.data.FastaBatchedDataset.from_file(args.query)
        # tune_esm_inference(data)
        # tune_esm_train(data, int(args.GPUs))
    
    else:    
        if args.logger == True:
            logger = TensorBoardLogger("logs")
        else:
            logger = False
        if args.profiler:
            profiler = PyTorchProfiler(filename='test-logs')
        else:
            profiler = None

    commands[args.command].run(args, logger, profiler)
    
    end = time.time()
    print("Finished!")
    print(f"Time elapsed: {end-start} seconds")
 

def cli(args=None):
    if not args:
        args = sys.argv[1:]    
    main(args)
if __name__ == '__main__':
    print("this shouldn't show up...")
