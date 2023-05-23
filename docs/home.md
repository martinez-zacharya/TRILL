                              _____________________.___.____    .____     
                              \__    ___/\______   \   |    |   |    |    
                                |    |    |       _/   |    |   |    |    
                                |    |    |    |   \   |    |___|    |___ 
                                |____|    |____|_  /___|_______ \_______ \
                                                 \/            \/       \/

[![version](https://img.shields.io/pypi/v/trill-proteins?color=blueviolet&style=flat-square)](https://pypi.org/project/trill-proteins)
![downloads](https://img.shields.io/pypi/dm/trill-proteins?color=blueviolet&style=flat-square)
[![license](https://img.shields.io/pypi/l/trill-proteins?color=blueviolet&style=flat-square)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/trill/badge/?version=latest&style=flat-square)](https://trill.readthedocs.io/en/latest/?badge=latest)
![status](https://github.com/martinez-zacharya/TRILL/workflows/CI/badge.svg?style=flat-square&color=blueviolet)
# Intro
TRILL (**TR**aining and **I**nference using the **L**anguage of **L**ife) is a sandbox for creative protein engineering and discovery. As a bioengineer myself, deep-learning based approaches for protein design and analysis are of great interest to me. However, many of these deep-learning models are rather unwieldy, especially for non ML-practitioners due to their sheer size. Not only does TRILL allow researchers to perform inference on their proteins of interest using a variety of models, but it also democratizes the efficient fine-tuning of large-language models. Whether using Google Colab with one GPU or a supercomputer with many, TRILL empowers scientists to leverage models with millions to billions of parameters without worrying (too much) about hardware constraints. Currently, TRILL supports using these models as of v1.3.0:
- ESM2 (Embed and Finetune all sizes, depending on hardware constraints [doi](https://doi.org/10.1101/2022.07.20.500902). Can also generate synthetic proteins from finetuned ESM2 models using Gibbs sampling [doi](https://doi.org/10.1101/2021.01.26.428322))
- ESM-IF1 (Generate synthetic proteins from .pdb backbone [doi](https://doi.org/10.1101/2022.04.10.487779))
- ESMFold (Predict 3D protein structure [doi](https://doi.org/10.1101/2022.07.20.500902))
- ProtGPT2 (Finetune and generate synthetic proteins from seed sequence [doi](https://doi.org/10.1038/s41467-022-32007-7))
- ProteinMPNN (Generate synthetic proteins from .pdb backbone [doi](https://doi.org/10.1101/2022.06.03.494563))
- RFDiffusion (Diffusion-based model for generating synthetic proteins [doi](https://doi.org/10.1101/2022.12.09.519842))
- DiffDock (Find best poses for protein-ligand binding [doi](https://doi.org/10.48550/arXiv.2210.01776))
- ProtT5-XL (Embed proteins into high-dimensional space [doi](https://doi.org/10.1109/TPAMI.2021.3095381))
- TemStaPro (Predict thermostability of proteins [doi](https://doi.org/10.1101/2023.03.27.534365))
- ZymCTRL (Conditional language model for the generation of artificial functional enzymes [link](https://www.mlsb.io/papers_2022/ZymCTRL_a_conditional_language_model_for_the_controllable_generation_of_artificial_enzymes.pdf))

## Set-Up
1. I recommend using a virtual environment with conda. If you don't have conda installed, follow these steps
```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh ./Miniconda3-latest-Linux-x86_64.sh
```
2. Once conda is set up, create a new environment with
```shell
conda create -n TRILL python=3.10
conda activate TRILL ; conda install pytorch==1.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia ; conda install -c conda-forge openbabel
```
3. Next, you need to download the Smina binary to perform binding affinity prediction. You should add smina to your path, because if not, you will need to have smina in the working directory wherever you run TRILL. This should hopefully be more smoothly integrated into TRILL soon
```shell
wget -O smina https://sourceforge.net/projects/smina/files/smina.static/download ; chmod +x smina
```
4. Next, simply install TRILL!
```shell
pip install trill-proteins
```

## Use

```
usage: trill [-h] [--nodes NODES] [--logger LOGGER] [--profiler] [--RNG_seed RNG_SEED]
             name GPUs
             {embed,finetune,inv_fold_gen,lang_gen,diff_gen,classify,fold,visualize,dock}
             ...

positional arguments:
  name                  Name of run
  GPUs                  Input total number of GPUs per node
  {embed,finetune,inv_fold_gen,lang_gen,diff_gen,classify,fold,visualize,dock}
    embed               Embed proteins of interest
    finetune            Finetune protein language models
    inv_fold_gen        Generate proteins using inverse folding
    lang_gen            Generate proteins using large language models including ProtGPT2
                        and ESM2
    diff_gen            Generate proteins using RFDiffusion
    classify            Classify proteins based on thermostability predicted through
                        TemStaPro
    fold                Predict 3D protein structures using ESMFold
    visualize           Reduce dimensionality of embeddings to 2D
    dock                Dock protein to protein using DiffDock

options:
  -h, --help            show this help message and exit
  --nodes NODES         Input total number of nodes. Default is 1
  --logger LOGGER       Enable Tensorboard logger. Default is None
  --profiler            Utilize PyTorchProfiler
  --RNG_seed RNG_SEED   Input RNG seed. Default is 123

```


## Examples

In the examples below the string immediately after `trill` specifies the name of the run. This name will be preprended to all outputs. For example, `trill example_1` will prepend `example_1` to the filenames that are generated by the run. The second argument specifies the number of GPUs to use in the run. If you don't have access to GPUs, you can simply put 0 to run TRILL on the CPU only.

### 1. Finetune Protein Language Models
  The default mode for TRILL is to just fine-tune the selected model with the query input for 20 epochs with a learning rate of 0.0001.
  ```
  trill example_1 1 finetune esm2_t12_35M trill/data/query.fasta
  ```
  By specifying --strategy, you can efficiently train large language models that would not normally be supported by your hardware. For example, if you run out of CUDA memory, you can try using Deepspeed or other strategies found [here](https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html). 
  ```
  trill example_1 1 finetune esm2_t36_3B trill/data/query.fasta --strategy deepspeed_stage_2
  ```
  You can finetune ProtGPT2.
  ```
  trill example_1 1 finetune ProtGPT2 trill/data/query.fasta
  ```
  You can also finetune ZymCTRL on certain a certain EC. Note that you must specify a EC tag that corresponds to ALL of the input proteins.
  ```
  trill example_1 1 finetune ZymCTRL trill/data/query.fasta --ctrl_tag 1.2.3.4
  ```
### 2. Create protein embeddings
  Use the embed command to create high-dimensional representations of your proteins of interest. Note that the model that produces the embeddings as well as an input protein fasta file is required.
  ```
  trill example_2 1 embed esm2_t12_35M trill/data/query.fasta
  ```  
  If you wanted to change the batch_size and use a finetuned ESM2 model for embeddings, you can specify it with --batch_size and --finetuned respectively
  ```
  trill example_2 1 embed esm2_t33_650M_UR50D trill/data/query.fasta --batch_size 2 --finetuned /path/to/models/finetuned_esm2_t30_150M_UR50D.pt
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
  
  srun trill example_3 4 finetune esm2_t33_650M_UR50D trill/data/query.fasta --nodes 4 --strategy deepspeed_stage_2_offload
  ```
  You can then submit this job with:
  ```
  sbatch distributed_example.slurm
  ```
  More examples for distributed training/inference without slurm coming soon!

### 4. Language Model Protein Generation
  You can use pretrained or finetuned protein language models such as ProtGPT2 or ESM2 to generate synthetic proteins with various hyperparameters.
  ProtGPT2: The command below generates 5 proteins. The default seed sequence is "M", but you can also change this. Check out the command-line arguments for more details.
  ```
  trill example_4 1 lang_gen ProtGPT2 --num_return_sequences 5
  ```
  In case you wanted to generate certain "types" of proteins with ProtGPT2, below is an example of using a fine-tuned ProtGPT2 to generate proteins.
  ```
  trill example_4 1 lang_gen ProtGPT2 --finetuned /path/to/FineTune_ProtGPT2_100.pt
  ```
  ESM2 Gibbs: Using Gibbs sampling, you can generate synthetic proteins from a finetuned ESM2 model. Note you must specify the ESM2 model architecture when doing gibbs sampling.
  ```
  trill example_4 1 lang_gen ESM2 --finetuned /path/to/finetuned_model.pt --esm2_arch esm2_t30_150M_UR50D --num_return_sequences 5
  ```
  ZymCTRL: By specifying an EC tag, you can control the type of enzyme the model tries to generate. If it is a class of enzymes that was not well represented in the ZymCTRL training set, you can first finetune it and then proceed to generate bespoke enzymes by passing the finetuned model with --finetuned.
  ```
  trill example_4 1 lang_gen ZymCTRL --num_return_sequences 5 --ctrl_tag 3.1.1.101
  ```
### 5. Inverse Folding Protein Generation
   ESM-IF1: When provided a protein backbone structure (.pdb, .cif), the IF1 model is able to predict a sequence that might be able to fold into the input structure backbone. The example input are the backbone coordinates from DWARF14, a rice hydrolase. For every chain in the structure, 2 in 4ih9.pdb, the following command will generate 3 sequences. In total, 6 sequences will be generated.
  ```
  trill example_5 1 inv_fold_gen ESM-IF1 trill/data/4ih9.pdb --num_return_sequences 3
  ```
  ProteinMPNN: Another model for inverse folding! You can specify the max length you want your protein to be with --max_length. Note that max_length must be at least as long as the input structure.
  ```
  trill example_5 1 inv_fold_gen ProteinMPNN trill/data/4ih9.pdb --max_length 1000 --num_return_sequences 5
  ```
### 6. Diffusion based Protein Generation
  RFDiffusion: You can perform a variety of protein design tasks, including designing binders! In the example below, you specify the target structure with --query, which in this case is an insulin receptor. The --contigs specify that we want residues 1-150 from chain A of the target structure, a chain break with /0 and a binder between 70-100 residues. We also specify residue hotspots, where we can tell the model to specifically target certain residues. Note that TRILL's implimentation of RFDiffusion does not yet have all of the knobs that the normal RFDiffusion has **yet**.
  ```
  trill example_6 1 diff_gen --query trill/data/insulin_target.pdb --contigs 'A1-150/0 70-100' --hotspots 'A59,A83,A91' --num_return_sequences 3
  ```
  More examples are coming soon for using RFDiffusion! I recommend checking out the examples for RFDiffusion on their [repo](https://github.com/RosettaCommons/RFdiffusion)
### 7. Predicting protein structure using ESMFold
  You can predict 3D protein structures rapidly in bulk using ESMFold. The output will be PDB files.
  ```
  trill example_7 1 fold trill/data/query.fasta
  ```  
### 8. Docking
  DiffDock: Currently, TRILL's implementation of DiffDock is only able to dock small molecules and proteins, but we are currently working on implementing DiffDock-PP for protein-protein docking. The output is ranked poses of the ligand.
  ```
  trill example_8 1 dock trill/data/4ih9.pdb trill/data/NAG_ideal.sdf
  ```
### 9. Visualize your embeddings
  Create interactive visualizations for your output embeddings in 2D. You can specify the dimensionality reduction method with --method.
  ```
  trill example_9 1 visualize /path/to/embeddings.csv
  ```  
## Misc. Tips

- Make sure there are no "\*" in the protein sequences
- After finetuning and trying to save a model using deepspeed, if all the CPU RAM is used the application can crash and not finish saving, leaving you a directory similar to "your_model.pt". You can rescue your model by running this python script
  ```  python
  from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
  convert_zero_checkpoint_to_fp32_state_dict(“your_model.pt”, “rescued_model.pt”)
  ```  
- If you are using TRILL on Google Colab, you need to start your commands with an "!".
  ```
  !trill example_10 1 embed esm2_t12_35M trill/data/query.fasta
  ```
