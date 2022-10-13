# DistantHomologyDetection

## Arguments

### Positional Arguments:
1. name (Name of run)
2. query (Input fasta file)
3. GPUs (Total # of GPUs requested for each node)

### Optional Arguments:
- -h, --help (Show help message)
- --database (Input database to embed with --blast mode)
- --nodes (Total number of computational nodes. Default is 1)
- --lr (Learning rate for adam optimizer. Default is 0.0001)
- --epochs (Number of epochs for fine-tuning transformer. Default is 20)
- --noTrain (Skips the fine-tuning and embeds the query sequences with the base model)
- --preTrained_model (Input path to your own pre-trained ESM model)
- --batch_size (Change batch-size number for fine-tuning. Default is 5)
- --blast (Enables "BLAST" mode. --database argument is required)
- --model (Change ESM model. Default is esm2_t12_35M_UR50D. List of models can be found at https://github.com/facebookresearch/esm)
- --strategy (Change training strategy. Default is None. List of strategies can be found at https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html)
- --logger (Enable Tensorboard logger. Default is None)

## Examples

### Default (Fine-tuning)
  1. The default mode for the pipeline is to just fine-tune the base esm1_t12 model from FAIR with the query input.
  ```
  python3 main.py fine_tuning_ex ../data/query.fasta 4
  ```
### Embed with base esm1_t12 model
  2. You can also embed proteins with just the base model from FAIR and completely skip fine-tuning.
  ```
  python3 main.py raw_embed ../data/query.fasta 4 --noTrain
  ```
### Embedding with a custom pre-trained model
  3. If you have a pre-trained model, you can use it to embed sequences by passing the path to --preTrained_model. 
  ```
  python3 main.py pre_trained ../data/query.fasta 4 --preTrained_model ../models/pre_trained_model.pt
  ```
### BLAST-like (Fine-tune on query and embed query+database)
  4. To enable a BLAST-like functionality, you can use the --blast flag in conjuction with passing a database fasta file to --database. The base model from FAIR is first fine-tuned with the query sequences and then both the query and the database sequences are embedded.
  ```
  python3 main.py blast_search ../data/query.fasta 4 --blast --database ../data/database.fasta
  ```
### Distributed Training/Inference
  5. In order to scale/speed up your analyses, you can distribute your training/inference across many GPUs with a few extra flags to your command. You can even fit models that do not normally fit on your GPUs with sharding and CPU-offloading. The list of strategies can be found here (https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html). The example below utilizes 16 GPUs in total (4(GPUs) * 4(--nodes)) with Fully Sharded Data Parallel and the 650M parameter ESM2 model.
  ```
  python3 main.py distributed_example ../data/query.fasta 4 --nodes 4 --strategy fsdp --model esm2_t33_650M_UR50D
  ```
  
## Quick Tutorial:

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
