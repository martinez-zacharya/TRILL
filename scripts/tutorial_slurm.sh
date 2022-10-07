#!/bin/bash
#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=2 # number of nodes
#SBATCH --gres=gpu:4 # number of GPUs (max 4 per node)
#SBATCH --mem-per-cpu=32G   # memory per CPU core
#SBATCH -J "test_lightning_base_esm1t12_8GPU"   # job name
#SBATCH --mail-user="zmartine@caltech.edu" # change to your email
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=%x-%j.out


module purge
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=13579
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_DEBUG=INFO
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/central/groups/mthomson/zam/miniconda3/lib
# export PL_IN_DDP_SUBPROCESS=1
export TORCH_HOME=/groups/mthomson/zam/.cache/torch/hub/checkpoints
source ~/.bashrc
# conda activate RemoteHomologyTransformer
conda install libgcc ;
pip install pandas ;
# conda install pytorch==1.12.0 cudatoolkit=11.3 python -c pytorch
# pip install torch==1.12.0 ;
# pip install fair-esm==1.0.2 ;
# pip install GPUtil ;
# pip install matplotlib;
# pip install Pillow ;
# pip install pytorch-lightning ;
# pip install deepspeed ;

# srun python3 main.py embed_archaea_base_esm1t12_16GPU ../data/0pt5_VP1s.fasta 1 --batch_size 5 --epochs 20
srun python3 newmain.py tetsing_lightning ../data/0pt5_VP1s.fasta 4 --batch_size 5 --epochs 5 --nodes 2 --strategy fsdp