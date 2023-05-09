                              _____________________.___.____    .____     
                              \__    ___/\______   \   |    |   |    |    
                                |    |    |       _/   |    |   |    |    
                                |    |    |    |   \   |    |___|    |___ 
                                |____|    |____|_  /___|_______ \_______ \
                                                 \/            \/       \/

[![pypi version](https://img.shields.io/pypi/v/trill-proteins?color=blueviolet&style=flat-square)](https://pypi.org/project/trill-proteins)
![downloads](https://img.shields.io/pypi/dm/trill-proteins?color=blueviolet&style=flat-square)
[![license](https://img.shields.io/pypi/l/trill-proteins?color=blueviolet&style=flat-square)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/trill/badge/?version=latest&style=flat-square)](https://trill.readthedocs.io/en/latest/?badge=latest)
![status](https://github.com/martinez-zacharya/TRILL/workflows/CI/badge.svg?style=flat-square&color=blueviolet)
# Intro
TRILL (**TR**aining and **I**nference using the **L**anguage of **L**ife) is a sandbox for creative protein engineering and discovery. As a bioengineer myself, deep-learning based approaches for protein design and analysis are of great interest to me. However, many of these deep-learning models are rather unwieldy, especially for non ML-practitioners due to their sheer size. Not only does TRILL allow researchers to perform inference on their proteins of interest using a variety of models, but it also democratizes the efficient fine-tuning of large-language models. Whether using Google Colab with one GPU or a supercomputer with many, TRILL empowers scientists to leverage models with millions to billions of parameters without worrying (too much) about hardware constraints. Currently, TRILL supports using these models as of v1.0.0:
- ESM2 (Embed and Finetune all sizes, depending on hardware constraints [doi](https://doi.org/10.1101/2022.07.20.500902). Can also generate synthetic proteins from finetuned ESM2 models using Gibbs sampling [doi](https://doi.org/10.1101/2021.01.26.428322))
- ESM-IF1 (Generate synthetic proteins from .pdb backbone [doi](https://doi.org/10.1101/2022.04.10.487779))
- ESMFold (Predict 3D protein structure [doi](https://doi.org/10.1101/2022.07.20.500902))
- ProtGPT2 (Finetune and generate synthetic proteins from seed sequence [doi](https://doi.org/10.1038/s41467-022-32007-7))
- ProteinMPNN (Generate synthetic proteins from .pdb backbone [doi](https://doi.org/10.1101/2022.06.03.494563))
- RFDiffusion (Diffusion-based model for generating synthetic proteins [doi](https://doi.org/10.1101/2022.12.09.519842))
- DiffDock (Find best poses for protein-ligand binding [doi](https://doi.org/10.48550/arXiv.2210.01776))
- ProtT5-XL (Embed proteins into high-dimensional space [doi](https://doi.org/10.1109/TPAMI.2021.3095381))
- TemStaPro (Predict thermostability of proteins [doi](https://doi.org/10.1101/2023.03.27.534365))

## Documentation
Check out the documentation and examples at https://trill.readthedocs.io/en/latest/index.html
