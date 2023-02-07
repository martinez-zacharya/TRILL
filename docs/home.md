                              _____________________.___.____    .____     
                              \__    ___/\______   \   |    |   |    |    
                                |    |    |       _/   |    |   |    |    
                                |    |    |    |   \   |    |___|    |___ 
                                |____|    |____|_  /___|_______ \_______ \
                                                 \/            \/       \/

 version](https://img.shields.io/pypi/v/trill-proteins?color=blueviolet&style=flat-square)](https://pypi.org/project/trill-proteins)
![downloads](https://img.shields.io/pypi/dm/trill-proteins?color=blueviolet&style=flat-square)
[![license](https://img.shields.io/pypi/l/trill-proteins?color=blueviolet&style=flat-square)](LICENSE)
![status](https://github.com/martinez-zacharya/TRILL/workflows/CI/badge.svg?style=flat-square&color=blueviolet)
[![Documentation Status](https://readthedocs.org/projects/trill/badge/?version=latest&style=flat-square)](https://trill.readthedocs.io/en/latest/?badge=latest)
# Intro
TRILL (**TR**aining and **I**nference using the **L**anguage of **L**ife) is a sandbox for creative protein engineering and discovery. As a bioengineer myself, deep-learning based approaches for protein design and analysis are of great interest to me. However, many of these deep-learning models are rather unwieldy, especially for non ML-practitioners due to their sheer size in parameters. Not only does TRILL allow researchers to perform inference on their proteins of interest using a variety of models, but it also democratizes the efficient fine-tuning of large-language models. Whether using Google Colab with one GPU or a supercomputer with many, TRILL empowers scientists to leverage models with millions to billions of parameters without worrying (too much) about hardware constraints. Currently, TRILL supports using these models as of v0.4.0:
- ESM2 (All sizes, depending on hardware constraints)
- ESM-IF1 (Generate synthetic proteins using Inverse-Folding)
- ESMFold (Predict 3D protein structure)
- ProtGPT2 (Generate synthetic proteins)

## Set-Up
1. I recommend using a virtual environment with conda, venv etc.
2. Run the following commands
```
$ pip install trill-proteins
```
```
$ pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

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
  
### 6. Visualize your embeddings
  Create interactive visualizations for your output embeddings in 2D. You can specify the dimensionality reduction method with --method.
  ```
  $ trill example_6 1 visualize /path/to/embeddings.csv
  ```  
## Misc. Tips

- Make sure there are no "\*" in the protein sequences
