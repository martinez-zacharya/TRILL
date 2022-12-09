                              _____________________.___.____    .____     
                              \__    ___/\______   \   |    |   |    |    
                                |    |    |       _/   |    |   |    |    
                                |    |    |    |   \   |    |___|    |___ 
                                |____|    |____|_  /___|_______ \_______ \
                                                 \/            \/       \/

# TRILL
**TR**aining and **I**nference using the **L**anguage of **L**ife

## Set-Up
1. Type ```git clone https://github.com/martinez-zacharya/TRILL``` to clone the repo
2. I recommend using a virtual environment with conda, venv etc.
3. Run ```pip install trill-proteins```
4. ```pip install torch```
5. ```pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html```

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
- --batch_size (Change batch-size number for fine-tuning. Default is 5)
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

## Examples

### Default (Fine-tuning)
  1. The default mode for TRILL is to just fine-tune the base esm2_t12_35M_UR50D model from FAIR with the query input.
  ```
  python3 trill.py fine_tuning_ex 1 --query data/query.fasta
  ```
### Embed with base esm2_t12_35M_UR50D model
  2. You can also embed proteins with just the base model from FAIR and completely skip fine-tuning.
  ```
  python3 trill.py base_embed 1 --query data/query.fasta --noTrain
  ```
### Embedding with a custom pre-trained model
  3. If you have a pre-trained model, you can use it to embed sequences by passing the path to --preTrained_model. 
  ```
  python3 trill.py pre_trained 1 --query data/query.fasta --preTrained_model /path/to/models/pre_trained_model.pt
  ```
### Distributed Training/Inference
  4. In order to scale/speed up your analyses, you can distribute your training/inference across many GPUs with a few extra flags to your command. You can even fit models that do not normally fit on your GPUs with sharding, CPU-offloading etc. Below is an example slurm batch submission file. The list of strategies can be found here (https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html). The example below utilizes 16 GPUs in total (4(GPUs) * 4(--nodes)) with Fully Sharded Data Parallel and the 650M parameter ESM2 model.
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
  
  srun python3 trill.py distributed_example 4 --query data/query.fasta --nodes 4 --strategy fsdp --model esm2_t33_650M_UR50D
  ```
  You can then submit this job with:
  ```
  sbatch distributed_example.slurm
  ```
  More examples for distributed training/inference without slurm coming soon!
  
### Generating protein sequences using inverse folding with ESM-IF1
  5. When provided a protein backbone structure (.pdb, .cif), the IF1 model is able to predict a sequence that might be able to fold into the input structure. The example input are the backbone coordinates from DWARF14, a rice hydrolase. For every chain in the structure, 2 in 4ih9.pdb, the following command will generate 3 sequences. In total, 6 sequences will be generated.
  ```
  python3 trill.py IF_Test 1 --query data/query.fasta --if1 --gen_iters 3
  ```
  
### Generating Proteins using ProtGPT2
  6. You can also generate synthetic proteins using ProtGPT2. The command below generates 5 proteins with a max length of 100. The default seed sequence is "M", but you can also change this. Check out the command-line arguments for more details.
  ```
  python3 trill.py Gen_ProtGPT2 1 --protgpt2 --gen --max_length 100 --num_return_sequences 5
  ```
  
### Fine-Tuning
  6. In case you wanted to generate certain "types" of proteins, below is an example of fine-tuning ProtGPT2 and then generating proteins with the fine-tuned model. 
  ```
  python3 trill.py FineTune 2 --protgpt2 --epochs 100
  ```
  ```
  python3 trill.py Gen_With_FineTuned 1 --protgpt2 --gen --preTrained_model FineTune_ProtGPT2_100.pt
  ```
  
## Quick Tutorial (NOT CURRENT, DON'T USE):

1. Type ```git clone https://github.com/martinez-zacharya/DistantHomologyDetection``` in your home directory on the HPC
2. Download Miniconda by running ```wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh``` and then ```sh ./Miniconda3-latest-Linux-x86_64.sh```.
3. Run ```conda env create -f environment.yml``` in the home directory of the repo to set up the proper conda environment and then type ```conda activate RemoteHomologyTransformer``` to activate it.
4. Shift your current working directory to the scripts folder with ```cd scripts```.
5. Type ```vi tutorial_slurm``` to open the slurm file and then hit ```i```.
6. Change the email in the tutorial_slurm file to your email (You can use https://s3-us-west-2.amazonaws.com/imss-hpc/index.html to make your own slurm files in the future).
7. Save the file by first hitting escape and then entering ```:x``` to exit and save the file. 
8. You can view the arguments for the command line tool by typing ```python3 main.py -h```.
9. To run the tutorial analysis, make the tutorial slurm file exectuable with ```chmod +x tutorial_slurm.sh``` and then type ```sbatch tutorial_slurm.sh```.
10. You can now safely exit the ssh instance to the HPC if you want

## Misc. Tips

- Make sure there are no "\*" in the protein sequences
- Don't run jobs on the login node, only submit jobs with sbatch or srun on the HPC
- Caltech HPC Docs https://www.hpc.caltech.edu/documentation
