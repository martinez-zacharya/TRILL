                              _____________________.___.____    .____     
                              \__    ___/\______   \   |    |   |    |    
                                |    |    |       _/   |    |   |    |    
                                |    |    |    |   \   |    |___|    |___ 
                                |____|    |____|_  /___|_______ \_______ \
                                                 \/            \/       \/

[![pypi version](https://img.shields.io/pypi/v/trill-proteins)](https://pypi.org/project/trill-proteins)
![status](https://github.com/martinez-zacharya/TRILL/workflows/CI/badge.svg)
# TRILL
**TR**aining and **I**nference using the **L**anguage of **L**ife

## Set-Up
1. I recommend using a virtual environment with conda, venv etc.
2. Run ```$ pip install trill-proteins```
3. ```$ pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html```

## Examples

### 1. Finetune
  The default mode for TRILL is to just fine-tune the base esm2_t12_35M_UR50D model from FAIR with the query input for 20 epochs with a learning rate of 0.0001.
  ```
  $ trill example_1 1 finetune data/query.fasta
  ```
  By specifying --model, you can change the model you want to finetune.
  ```
  $ trill example_1 1 finetune data/query.fasta --model esm2_t30_150M_UR50D
  ```
  You can also finetune ProtGPT2
  ```
  $ trill example_1 1 finetune data/query.fasta --model ProtGPT2
  ```
### 2. Create protein embeddings
  Using the embed command by default uses esm2_t12_35M_UR50D to create high-dimensional representations of your proteins of interest.
  ```
  $ trill example_2 1 embed data/query.fasta
  ```  
  If you wanted to use another ESM2 model and change the batch_size, you can specify it with --model and --batch_size respectively
  ```
  $ trill example_2 1 embed data/query.fasta --model esm2_t33_650M_UR50D --batch_size 5
  ```
  To use a custom finetuned ESM2 model for embeddings, you can pass the path to --preTrained_model. Make sure to include what the base model was for your finetuned model with --model
  ```
  $ trill example_2 1 data/query.fasta --preTrained_model /path/to/models/finetuned_esm2_t30_150M_UR50D.pt --model esm2_t30_150M_UR50D
  ```
### 3. Distributed Training/Inference
  In order to scale/speed up your analyses, you can distribute your training/inference across many GPUs with a few extra flags to your command. You can even fit models that do not normally fit on your GPUs with sharding, CPU-offloading etc. Below is an example slurm batch submission file. The list of strategies can be found here (https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html). The example below utilizes 16 GPUs in total (4(GPUs) * 4(--nodes)) with deepspeed_stage_2_offload and the 650M parameter ESM2 model.
  ```shell
  #!/bin/bash
  #SBATCH --time=8:00:00   # walltime
  #SBATCH --ntasks-per-node=4
  #SBATCH --nodes=4 # number of nodes
  #SBATCH --gres=gpu:4 # number of GPUs
  #SBATCH --mem-per-cpu=60G   # memory per CPU core
  #SBATCH -J "tutorial"   # job name
  #SBATCH --mail-user="" # change to your email
  #SBATCH --mail-type=BEGIN
  #SBATCH --mail-type=END
  #SBATCH --mail-type=FAIL
  #SBATCH --output=%x-%j.out
  master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  export MASTER_ADDR=$master_addr
  export MASTER_PORT=13579
  
  srun trill example_3 4 finetune data/query.fasta --nodes 4 --strategy deepspeed_stage_2_offload --model esm2_t33_650M_UR50D
  ```
  You can then submit this job with:
  ```
  $ sbatch distributed_example.slurm
  ```
  More examples for distributed training/inference without slurm coming soon!

### 4. Generate synthetic proteins
   ESM-IF1: When provided a protein backbone structure (.pdb, .cif), the IF1 model is able to predict a sequence that might be able to fold into the input structure. The example input are the backbone coordinates from DWARF14, a rice hydrolase. For every chain in the structure, 2 in 4ih9.pdb, the following command will generate 3 sequences. In total, 6 sequences will be generated.
  ```
  $ trill example_4 1 generate ESM-IF1 --query data/4ih9.pdb --genIters 3
  ```
  You can also generate synthetic proteins using ProtGPT2. The command below generates 5 proteins with a max length of 100. The default seed sequence is "M", but you can also change this. Check out the command-line arguments for more details.
  ```
  $ trill example_4 1 generate ProtGPT2 --max_length 100 --num_return_sequences 5
  ```
  In case you wanted to generate certain "types" of proteins, below is an example of using a fine-tuned ProtGPT2 to generate proteins.
  ```
  $ trill example_4 1 generate ProtGPT2 --finetuned_protgpt2 /path/to/FineTune_ProtGPT2_100.pt
  ```
### 5. Predicting protein structure using ESMFold
  You can predict 3D protein structures rapidly in bulk using ESMFold. The output will be PDB files.
  ```
  $ trill example_5 1 fold data/query.fasta
  ```  

## Arguments

### Positional Arguments:
1. name (Name of run)
2. GPUs (Total # of GPUs requested for each node)

### Optional Arguments:
- -h, --help (Show help message)
- --query (Input file. Needs to be either protein fasta (.fa, .faa, .fasta) or structural coordinates (.pdb, .cif))
- --nodes (Total number of computational nodes. Default is 1)
- --lr (Learning rate for adam optimizer. Default is 0.0001)
- --epochs (Number of epochs for fine-tuning transformer. Default is 20)
- --noTrain (Skips the fine-tuning and embeds the query sequences with the base model)
- --preTrained_model (Input path to your own pre-trained ESM model)
- --batch_size (Change batch-size number for fine-tuning. Default is 1)
- --model (Change ESM model. Default is esm2_t12_35M_UR50D. List of models can be found at https://github.com/facebookresearch/esm)
- --strategy (Change training strategy. Default is None. List of strategies can be found at https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html)
- --logger (Enable Tensorboard logger. Default is None)
- --if1 (Utilize Inverse Folding model 'esm_if1_gvp4_t16_142M_UR50' to facilitate fixed backbone sequence design. Basically converts protein structure to possible sequences)
- --temp (Choose sampling temperature. Higher temps will have more sequence diversity, but less recovery of the original sequence for ESM_IF1)
- --genIters (Adjust number of sequences generated for each chain of the input structure for ESM_IF1)
- --LEGGO (Use deepspeed_stage_3_offload with ESM. Will be removed soon...)
- --profiler (Utilize PyTorchProfiler)
- --protgpt2 (Utilize ProtGPT2. Can either fine-tune or generate sequences)
- --gen (Generate protein sequences using ProtGPT2. Can either use base model or user-submitted fine-tuned model)
- --seed_seq (Sequence to seed ProtGPT2 Generation)
- --max_length (Max length of proteins generated from ProtGPT)
- --do_sample (Whether or not to use sampling ; use greedy decoding otherwise)
- --top_k (The number of highest probability vocabulary tokens to keep for top-k-filtering)
- --repetition_penalty (The parameter for repetition penalty. 1.0 means no penalty)
- --num_return_sequences (Number of sequences for ProtGPT2 to generate)
- --esmfold (Predict protein structures using ESMFold)

## Misc. Tips

- Make sure there are no "\*" in the protein sequences
- Don't run jobs on the login node, only submit jobs with sbatch or srun on the HPC
- Caltech HPC Docs https://www.hpc.caltech.edu/documentation
