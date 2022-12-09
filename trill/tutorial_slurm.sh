#!/bin/bash
#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1 # number of nodes
#SBATCH --gres=gpu:4 # number of GPUs (max 4 per node)
#SBATCH --mem-per-cpu=60G   # memory per CPU core
#SBATCH -J "tutorial"   # job name
#SBATCH --mail-user="" # change to your email
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=%x-%j.out


module purge
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=13579
export NCCL_SOCKET_IFNAME=^docker0,lo
# export NCCL_DEBUG=INFO
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/central/groups/mthomson/zam/miniconda3/lib
export TORCH_HOME=/groups/mthomson/zam/.cache/torch/hub/checkpoints
source ~/.bashrc

srun python3 newmain.py tutorial ../data/query.fasta 4 --epochs 5 --strategy fsdp
