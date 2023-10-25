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
<!-- ![status](https://github.com/martinez-zacharya/TRILL/workflows/CI/badge.svg?style=flat-square&color=blueviolet) -->
# Intro
TRILL (**TR**aining and **I**nference using the **L**anguage of **L**ife) is a sandbox for creative protein engineering and discovery. As a bioengineer myself, deep-learning based approaches for protein design and analysis are of great interest to me. However, many of these deep-learning models are rather unwieldy, especially for non ML-practitioners due to their sheer size. Not only does TRILL allow researchers to perform inference on their proteins of interest using a variety of models, but it also democratizes the efficient fine-tuning of large-language models. Whether using Google Colab with one GPU or a supercomputer with many, TRILL empowers scientists to leverage models with millions to billions of parameters without worrying (too much) about hardware constraints. Currently, TRILL supports using these models as of v1.5.0:

## Breakdown of TRILL's Commands

| **Command** | **Function** | **Available Models** |
|:-----------:|:------------:|:--------------------:|
| **Embed** | Generates numerical representations or "embeddings" of protein sequences for quantitative analysis and comparison. | [ESM2](https://doi.org/10.1101/2022.07.20.500902), [ProtT5-XL](https://doi.org/10.1109/TPAMI.2021.3095381), [ProstT5](https://doi.org/10.1101/2023.07.23.550085) |
| **Visualize** | Creates interactive 2D visualizations of embeddings for exploratory data analysis. | PCA, t-SNE, UMAP |
| **Finetune** | Finetunes protein language models for specific tasks. | [ESM2](https://doi.org/10.1101/2022.07.20.500902), [ProtGPT2](https://doi.org/10.1038/s41467-022-32007-7), [ZymCTRL](https://www.mlsb.io/papers_2022/ZymCTRL_a_conditional_language_model_for_the_controllable_generation_of_artificial_enzymes.pdf) |
| **Language Model Protein Generation** | Generates proteins using pretrained language models. | [ESM2](https://doi.org/10.1101/2022.07.20.500902), [ProtGPT2](https://doi.org/10.1038/s41467-022-32007-7), [ZymCTRL](https://www.mlsb.io/papers_2022/ZymCTRL_a_conditional_language_model_for_the_controllable_generation_of_artificial_enzymes.pdf) |
| **Inverse Folding Protein Generation** | Designs proteins to fold into specific 3D structures. | [ESM-IF1](https://doi.org/10.1101/2022.04.10.487779), [ProteinMPNN](https://doi.org/10.1101/2022.06.03.494563), [ProstT5](https://doi.org/10.1101/2023.07.23.550085) |
| **Diffusion Based Protein Generation** | Uses denoising diffusion models to generate proteins. | [RFDiffusion](https://doi.org/10.1101/2022.12.09.519842) |
| **Fold** | Predicts 3D protein structures. | [ESMFold](https://doi.org/10.1101/2022.07.20.500902), [ProstT5](https://doi.org/10.1101/2023.07.23.550085) |
| **Dock** | Simulates protein-ligand interactions. | [DiffDock](https://doi.org/10.48550/arXiv.2210.01776), [Smina](https://doi.org/10.1021/ci300604z), [Autodock Vina](https://doi.org/10.1021/acs.jcim.1c00203), [Lightdock](https://doi.org/10.1093/bioinformatics/btx555) |
| **Classify** | Predicts protein properties at high throughput. | [TemStaPro](https://doi.org/10.1101/2023.03.27.534365), [EpHod](https://doi.org/10.1101/2023.06.22.544776) |
| **Simulate** | Uses molecular dynamics with the AMBER force field to relax structures. | [OpenMM](https://doi.org/10.1371/journal.pcbi.1005659) |



## Set-Up
1. TRILL has only ever been tested on Linux machines, but it might work on Windows with WSL2. You can use TRILL with Google Colab by using the installation instructions from [this notebook](https://colab.research.google.com/drive/1mx16cDAEgCYtflKm80mE8L_ZEk7Upxxe#scrollTo=AunNpGWa8tGn). For regular installations, I recommend using micromamba or mamba. If you don't have mamba installed, use this command
```shell
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
2. Once micromamba is set up, create a new environment with
```shell
micromamba create -n TRILL python=3.10 ; micromamba activate TRILL
micromamba install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge openmm swig pdbfixer openbabel smina fpocket vina -c bioconda pyrsistent foldseek -c pyg pyg=2.3.1=py310_torch_1.13.0_cu117 pytorch-cluster=1.6.1=py310_torch_1.13.0_cu117 pytorch-sparse=0.6.17=py310_torch_1.13.0_cu117 pytorch-scatter=2.1.1=py310_torch_1.13.0_cu117
micromamba install -c bioconda-c "dglteam/label/cu117" dgl
pip install git+https://github.com/martinez-zacharya/lightdock.git@03a8bc4888c0ff8c98b7f0df4b3c671e3dbf3b1f
```
3. Next, simply install TRILL!
```shell
pip install trill-proteins
```

## Use

```shell
usage: trill [-h] [--nodes NODES] [--logger LOGGER] [--profiler] [--RNG_seed RNG_SEED] [--outdir OUTDIR] [--n_workers N_WORKERS]
             name GPUs {embed,finetune,inv_fold_gen,lang_gen,diff_gen,classify,fold,visualize,simulate,dock,utils} ...

positional arguments:
  name                  Name of run
  GPUs                  Input total number of GPUs per node
  {embed,finetune,inv_fold_gen,lang_gen,diff_gen,classify,fold,visualize,simulate,dock,utils}
    embed               Embed proteins of interest
    finetune            Finetune protein language models
    inv_fold_gen        Generate proteins using inverse folding
    lang_gen            Generate proteins using large language models
    diff_gen            Generate proteins using RFDiffusion
    classify            Classify proteins based on thermostability predicted through TemStaPro
    fold                Predict 3D protein structures using ESMFold or obtain 3Di structure for use with Foldseek to perform remote homology detection
    visualize           Reduce dimensionality of embeddings to 2D
    simulate            Use MD to relax protein structures
    dock                Perform molecular docking with proteins and ligands. Note that you should relax your protein receptor with Simulate or another method before docking.
    utils               Misc utilities

options:
  -h, --help            show this help message and exit
  --nodes NODES         Input total number of nodes. Default is 1
  --logger LOGGER       Enable Tensorboard logger. Default is None
  --profiler            Utilize PyTorchProfiler
  --RNG_seed RNG_SEED   Input RNG seed. Default is 123
  --outdir OUTDIR       Input full path to directory where you want the output from TRILL
  --n_workers N_WORKERS
                        Change number of CPU cores/'workers' TRILL uses


```


## Examples

In the examples below the string immediately after `trill` specifies the name of the run. This name will be preprended to all outputs. For example, `trill example` will prepend `example` to the filenames that are generated by the run. The second argument specifies the number of GPUs to use in the run. If you don't have access to GPUs, you can simply put 0 to run TRILL on the CPU only.

### 1. Finetune Protein Language Models
  The default mode for TRILL is to just fine-tune the selected model with the query input for 10 epochs with a learning rate of 0.0001.
  ```
  trill example 1 finetune esm2_t12_35M trill/data/query.fasta
  ```
  By specifying --strategy, you can efficiently train large language models that would not normally be supported by your hardware. For example, if you run out of CUDA memory, you can try using Deepspeed or other strategies found [here](https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html). 
  ```
  trill example 1 finetune esm2_t36_3B trill/data/query.fasta --strategy deepspeed_stage_2
  ```
  You can finetune ProtGPT2.
  ```
  trill example 1 finetune ProtGPT2 trill/data/query.fasta
  ```
  You can also finetune ZymCTRL on certain a certain EC. Note that you must specify a EC tag that corresponds to ALL of the input proteins.
  ```
  trill example 1 finetune ZymCTRL trill/data/query.fasta --ctrl_tag 1.2.3.4
  ```
### 2. Create protein embeddings
  Use the embed command to create high-dimensional representations of your proteins of interest. --avg returns the averaged, whole sequence embeddings, while --per_AA returns the per amino acid representation for each AA in each sequence.
  ```
  trill example 1 embed esm2_t12_35M trill/data/query.fasta --avg --per_AA
  ```  
  If you wanted to change the batch_size and use a finetuned ESM2 model for embeddings, you can specify it with --batch_size and --finetuned respectively. Also note that you can opt to only return one type of representation.
  ```
  trill example 1 embed esm2_t33_650M trill/data/query.fasta --batch_size 2 --finetuned /path/to/models/finetuned_esm2_t30_150M_UR50D.pt --avg
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
  
  srun trill example 4 finetune esm2_t33_650M trill/data/query.fasta --nodes 4 --strategy deepspeed_stage_3_offload
  ```
  You can then submit this job with:
  ```
  sbatch distributed_example.slurm
  ```
  More examples for distributed training/inference without slurm coming soonish!

### 4. Language Model Protein Generation
  You can use pretrained or finetuned protein language models to generate synthetic proteins with various hyperparameters.
  ProtGPT2: The command below generates 5 proteins. The default seed sequence is "M", but you can also change this. Check out the command-line arguments for more details.
  ```
  trill example 1 lang_gen ProtGPT2 --num_return_sequences 5
  ```
  In case you wanted to generate certain "types" of proteins with ProtGPT2, below is an example of using a fine-tuned ProtGPT2 to generate proteins.
  ```
  trill example 1 lang_gen ProtGPT2 --finetuned /path/to/FineTune_ProtGPT2_100.pt
  ```
  ESM2 Gibbs: Using Gibbs sampling, you can generate synthetic proteins from a finetuned ESM2 model. Note you must specify the ESM2 model architecture when doing gibbs sampling.
  ```
  trill example 0 lang_gen ESM2 --finetuned /path/to/finetuned_model.pt --esm2_arch esm2_t30_150M_UR50D --num_return_sequences 5
  ```
  ZymCTRL: By specifying an EC tag, you can control the type of enzyme the model tries to generate. If it is a class of enzymes that was not well represented in the ZymCTRL training set, you can first finetune it and then proceed to generate bespoke enzymes by passing the finetuned model with --finetuned.
  ```
  trill example 1 lang_gen ZymCTRL --num_return_sequences 5 --ctrl_tag 3.1.1.101
  ```
### 5. Inverse Folding Protein Generation
   ESM-IF1: When provided a protein structure the IF1 model is able to predict a sequence that might be able to fold into the input structure backbone. The example input are the backbone coordinates from DWARF14, a rice hydrolase. For every chain in the structure, 2 in 4ih9.pdb, the following command will generate 3 sequences. In total, 6 sequences will be generated.
  ```
  trill example 1 inv_fold_gen ESM-IF1 trill/data/4ih9.pdb --num_return_sequences 3
  ```
  ProteinMPNN: You can specify the max length you want your protein to be with --max_length. Note that max_length must be at least as long as the input structure.
  ```
  trill example 1 inv_fold_gen ProteinMPNN trill/data/4ih9.pdb --max_length 1000 --num_return_sequences 5
  ```
  ProstT5: This bilingual protein language model is technically able to perform inverse folding. First, TRILL uses foldseek to extract 3Di tokens from the .pdb, and then ProstT5 translates the tokens into potential sequences. With this model, you can set hyperparameters such as --top_p in order to perform nucleus sampling and only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
  ```
  trill example 1 inv_fold_gen ProstT5 trill/data/4ih9.pdb --max_length 1000 --num_return_sequences 5 --top_p 0.9
  ```
### 6. Diffusion based Protein Generation
  RFDiffusion: You can perform a variety of protein design tasks, including designing binders! In the example below, you specify the target structure with --query, which in this case is an insulin receptor. The --contigs specify that we want residues 1-150 from chain A of the target structure, a chain break with /0 and a binder between 70-100 residues. We also specify residue hotspots, where we can tell the model to specifically target certain residues. Note that TRILL's implimentation of RFDiffusion does not yet have all of the knobs that the normal RFDiffusion has **yet**.
  ```
  trill example 1 diff_gen --query trill/data/insulin_target.pdb --contigs 'A1-150/0 70-100' --hotspots 'A59,A83,A91' --num_return_sequences 3
  ```
  More examples are coming soon for using RFDiffusion! I recommend checking out the examples for RFDiffusion on their [repo](https://github.com/RosettaCommons/RFdiffusion)
### 7. Predicting protein structure
  You can predict 3D protein structures rapidly in bulk using ESMFold. The output will be PDB files.
  ```
  trill example 1 fold ESMFold trill/data/query.fasta
  ```
  While not technically returning a 3D structure, ProstT5 is able to predict 3Di tokens from sequence alone, which can then be used with Foldseek!
  ```
  trill example 1 fold ProstT5 trill/data/query.fasta
  ```
### 8. Docking
  DiffDock: Uses deep-learning to dock small-molecules to proteins. The output is ranked poses of the ligand.
  ```
  trill example 1 dock DiffDock trill/data/4ih9.pdb trill/data/NAG_ideal.sdf
  ```
  Lightdock: Dock proteins to proteins using glowworm swarm optimization, with support for docking proteins to DNA/RNA coming soon! 
  ```
  trill example 1 dock Lightdock trill/data/4ih9.pdb trill/data/peptide.pdb
  ```
  Autodock Vina: Dock small-molecule(s) to a protein. Not only can you perform blind docking, but with Vina exclusively in TRILL, you are able to dock multiple ligands at once!
  ```
  trill example 1 dock Lightdock trill/data/4ih9.pdb ligand_1.sdf ligand_2.sdf --blind
  ```
### 9. Visualize your embeddings
  Create interactive, queryable visualizations for your output embeddings in 2D. TRILL uses PCA by default, but you can specify tSNE or UMAP with --method.
  ```
  trill example 1 visualize /path/to/embeddings.csv
  ```
### 10. Relax protein structure(s) using molecular dynamics.
  Using OpenMM, TRILL is able to relax protein structures, which is often needed before performing docking. Be on the lookout for more MD related features!
  ```
  trill example 1 simulate /path/to/embeddings.csv
  ```
### 11. Leverage pretrained classifiers or train your own custom ones.
  Currently, TRILL offers two pretrained protein classifiers, EpHod and TemStaPro, which predict optimal enzymatic pH and thermostability respectively. The subsequent predictions of the models are saved as a csv.
  ```
  trill example 1 classify EpHod trill/data/query.fasta
  ```
  TRILL also allows users to train custom XGBoost classifiers or Isolation Forest anomaly detectors with the average sequence representation extracted with a protein language model. While XGBoost benefits from a balanced training set, the iForest only needs one label and it predicts whether a protein is an anomaly or not. To train a model, you only need to provide protein sequences in fasta form, as well as a csv that contains the class mappings. First, you can use the utils command to prepare the class key by either passing a directory with --dir and your desired classes should be separated into different fasta files according to their class. If there are 5 fasta files, there will be 5 classes. You can also pass a .txt file with absolute paths to fasta files separated by new lines and every fasta file will be treated as a unique class.
  ```
  trill example 0 utils prepare_class_key --fasta_paths_txt my_classes.txt
  ```
  Once you have your key, you can train your custom XGBoost model! TRILL automatically partitions your sequences into training and validation sets, you can adjust the ratio with --train_split. After training, the model is evaluated on a held-out validation set and metrics such as F-score are calculated.
  ```
  trill example 1 classify XGBoost train_master.fasta --train_split .8 --key my_key.csv 
  ```
  The XGBoost model will be saved as a .json, which can be loaded with --preTrained to predict the classes of new sequences. Note that if you have already embedded your sequences, you can pass the csv with --preComputed_Embs and the corresponding model that was used to create the embeddings with --emb_model. The preComputed_Embs must be extracted from the same model that the XGBoost model was trained on to begin with. Regardless, the predictions will be saved as a csv. 
  ```
  trill example 1 classify XGBoost test_master.fasta --preTrained my_xgboost.json
  ```
  You might find yourself in a situation where you have a protein classification of interest, but there isn't an obvious way to find negative examples of this classification. Using just one type of label, you can train an Isolation Forest to predict whether new sequences are anomalies. 
  ```
  trill example 1 classify iForest train.fasta --emb_model esm2_t30_150M
  trill example 1 classify iForest test.fasta --emb_model esm2_t30_150M --preTrained my_iforest.skops
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
  !trill example 1 embed esm2_t12_35M trill/data/query.fasta
  ```
- If you recieve this error when trying to use deepspeed CPU offloading
  ```
  AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'
  ```
  you can potentially fix the issue by using mamba or micromamba to install -c "nvidia/label/cuda-11.7.0" cuda-nvcc
