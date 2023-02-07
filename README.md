                              _____________________.___.____    .____     
                              \__    ___/\______   \   |    |   |    |    
                                |    |    |       _/   |    |   |    |    
                                |    |    |    |   \   |    |___|    |___ 
                                |____|    |____|_  /___|_______ \_______ \
                                                 \/            \/       \/

[![pypi version](https://img.shields.io/pypi/v/trill-proteins?color=blueviolet&style=flat-square)](https://pypi.org/project/trill-proteins)
![downloads](https://img.shields.io/pypi/dm/trill-proteins?color=blueviolet&style=flat-square)
[![license](https://img.shields.io/pypi/l/trill-proteins?color=blueviolet&style=flat-square)](LICENSE)
![status](https://github.com/martinez-zacharya/TRILL/workflows/CI/badge.svg?style=flat-square&color=blueviolet)
[![Documentation Status](https://readthedocs.org/projects/trill/badge/?version=latest&style=flat-square)](https://trill.readthedocs.io/en/latest/?badge=latest)
# TRILL
**TR**aining and **I**nference using the **L**anguage of **L**ife is a sandbox for creative protein engineering and discovery. As a bioengineer myself, deep-learning based approaches for protein design and analysis are of great interest to me. However, many of these deep-learning models are rather unwieldy, especially for non ML-practitioners, due to their sheer size. Not only does TRILL allow researchers to perform inference on their proteins of interest using a variety of models, but it also democratizes the efficient fine-tuning of large-language models. Whether using Google Colab with one GPU or a supercomputer with many, TRILL empowers scientists to leverage models with millions to billions of parameters without worrying (too much) about hardware constraints. Currently, TRILL supports using these models as of v0.4.0:
- ESM2 (All sizes, depending on hardware constraints [doi](https://doi.org/10.1101/2022.07.20.500902))
- ESM-IF1 (Generate synthetic proteins using from .pdb backbone [doi](https://doi.org/10.1101/2022.04.10.487779))
- ESMFold (Predict 3D protein structure [doi](https://doi.org/10.1101/2022.07.20.500902))
- ProtGPT2 (Generate synthetic proteins from seed sequence [doi](https://doi.org/10.1038/s41467-022-32007-7))
- ProteinMPNN (Generate synthetic proteins from .pdb backbone [doi](https://doi.org/10.1101/2022.06.03.494563))

## Documentation
Check out the documentation and examples at https://trill.readthedocs.io/en/latest/home.html
