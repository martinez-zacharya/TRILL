                              _____________________.___.____    .____     
                              \__    ___/\______   \   |    |   |    |    
                                |    |    |       _/   |    |   |    |    
                                |    |    |    |   \   |    |___|    |___ 
                                |____|    |____|_  /___|_______ \_______ \
                                                 \/            \/       \/

[![pypi version](https://img.shields.io/pypi/v/trill-proteins)](https://pypi.org/project/trill-proteins)
![status](https://github.com/martinez-zacharya/TRILL/workflows/CI/badge.svg)
# TRILL
**TR**aining and **I**nference using the **L**anguage of **L**ife is a sandbox for creative protein engineering and discovery. As a bioengineer myself, deep-learning based approaches for protein design and analysis are of great interest to me. However, many of these deep-learning models are rather unwieldy, especially for non ML-practitioners, due to their sheer size. Not only does TRILL allow researchers to perform inference on their proteins of interest using a variety of models, but it also democratizes the efficient fine-tuning of large-language models. Whether using Google Colab with one GPU or a supercomputer with many, TRILL empowers scientists to leverage models with millions to billions of parameters without worrying (too much) about hardware constraints. Currently, TRILL supports using these models as of v0.4.0:
- ESM2 (All sizes, depending on hardware constraints)
- ESM-IF1 (Generate synthetic proteins using Inverse-Folding)
- ESMFold (Predict 3D protein structure)
- ProtGPT2 (Generate synthetic proteins)

## Documentation
Check out the documentation and examples at https://trill.readthedocs.io/en/latest/home.html
