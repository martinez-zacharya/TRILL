#!/bin/bash
#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:4 # number of GPUs (max 4 per node)
#SBATCH --mem-per-cpu=32G   # memory per CPU core
#SBATCH -J "Fine_Tuning_with X Protein Family"   # job name
#SBATCH --mail-user="user@caltech.edu " # change to your email
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=%x-%j.out


module purge
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
# export NCCL_DEBUG=INFO
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib
source ~/.bashrc
conda activate RemoteHomologyTransformer
conda install libgcc ;
pip install torch==1.12.0 ;

srun python3 main.py tutorial_run ../data/query.fasta 4 --epochs 5