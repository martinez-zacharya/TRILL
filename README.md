                              _____________________.___.____    .____     
                              \__    ___/\______   \   |    |   |    |    
                                |    |    |       _/   |    |   |    |    
                                |    |    |    |   \   |    |___|    |___ 
                                |____|    |____|_  /___|_______ \_______ \
                                                 \/            \/       \/

[![pypi version](https://img.shields.io/pypi/v/trill-proteins?color=blueviolet&style=flat-square)](https://pypi.org/project/trill-proteins)
![Downloads](https://pepy.tech/badge/trill-proteins)
[![license](https://img.shields.io/pypi/l/trill-proteins?color=blueviolet&style=flat-square)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/trill/badge/?version=latest&style=flat-square)](https://trill.readthedocs.io/en/latest/?badge=latest)
<!---![status](https://github.com/martinez-zacharya/TRILL/workflows/CI/badge.svg?style=flat-square&color=blueviolet)--->
# Intro
TRILL (**TR**aining and **I**nference using the **L**anguage of **L**ife) is a sandbox for creative protein engineering and discovery. As a bioengineer myself, deep-learning based approaches for protein design and analysis are of great interest to me. However, many of these deep-learning models are rather unwieldy, especially for non ML-practitioners due to their sheer size. Not only does TRILL allow researchers to perform inference on their proteins of interest using a variety of models, but it also democratizes the efficient fine-tuning of large-language models. Whether using Google Colab with one GPU or a supercomputer with many, TRILL empowers scientists to leverage models with millions to billions of parameters without worrying (too much) about hardware constraints. Currently, TRILL supports using these models as of v1.8.0:

## Breakdown of TRILL's Commands

| **Command** | **Function** | **Available Models** |
|:-----------:|:------------:|:--------------------:|
| **Embed** | Generates numerical representations or "embeddings" of protein sequences for quantitative analysis and comparison. | [ESM2](https://doi.org/10.1101/2022.07.20.500902), [ProtT5-XL](https://doi.org/10.1109/TPAMI.2021.3095381), [ProstT5](https://doi.org/10.1101/2023.07.23.550085), [Ankh](https://doi.org/10.48550/arXiv.2301.06568)|
| **Visualize** | Creates interactive 2D visualizations of embeddings for exploratory data analysis. | PCA, t-SNE, UMAP |
| **Finetune** | Finetunes protein language models for specific tasks. | [ESM2](https://doi.org/10.1101/2022.07.20.500902), [ProtGPT2](https://doi.org/10.1038/s41467-022-32007-7), [ZymCTRL](https://www.mlsb.io/papers_2022/ZymCTRL_a_conditional_language_model_for_the_controllable_generation_of_artificial_enzymes.pdf) |
| **Language Model Protein Generation** | Generates proteins using pretrained language models. | [ESM2](https://doi.org/10.1101/2022.07.20.500902), [ProtGPT2](https://doi.org/10.1038/s41467-022-32007-7), [ZymCTRL](https://www.mlsb.io/papers_2022/ZymCTRL_a_conditional_language_model_for_the_controllable_generation_of_artificial_enzymes.pdf) |
| **Inverse Folding Protein Generation** | Designs proteins to fold into specific 3D structures. | [ESM-IF1](https://doi.org/10.1101/2022.04.10.487779), [LigandMPNN](https://doi.org/10.1101/2023.12.22.573103), [ProstT5](https://doi.org/10.1101/2023.07.23.550085) |
| **Diffusion Based Protein Generation** | Uses denoising diffusion models to generate proteins. | [RFDiffusion](https://doi.org/10.1101/2022.12.09.519842) |
| **Fold** | Predicts 3D protein structures. | [ESMFold](https://doi.org/10.1101/2022.07.20.500902), [ProstT5](https://doi.org/10.1101/2023.07.23.550085) |
| **Dock** | Simulates protein-ligand interactions. | [DiffDock](https://doi.org/10.48550/arXiv.2210.01776), [Smina](https://doi.org/10.1021/ci300604z), [Autodock Vina](https://doi.org/10.1021/acs.jcim.1c00203), [Lightdock](https://doi.org/10.1093/bioinformatics/btx555), [GeoDock](https://doi.org/10.1101/2023.06.29.547134) |
| **Classify** | Predicts protein properties with pretrained models or train custom classifiers | [TemStaPro](https://doi.org/10.1101/2023.03.27.534365), [EpHod](https://doi.org/10.1101/2023.06.22.544776), [ECPICK](https://github.com/datax-lab/ECPICK?tab=readme-ov-file), [LightGBM](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), [XGBoost](https://doi.org/10.48550/arXiv.1603.02754), [Isolation Forest](https://doi.org/10.1109/ICDM.2008.17) |
| **Regress** | Train custom regression models. | [LightGBM](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), [Linear](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)|
| **Simulate** | Uses molecular dynamics to simulate protein-ligand interactions. | [OpenMM](https://doi.org/10.1371/journal.pcbi.1005659) |
| **Score** | Utilize ESM1v or ESM2 to score protein sequences or ProteinMPNN to score protein structures in a zero-shot manner. | [COMPSS](https://www.nature.com/articles/s41587-024-02214-2#change-history)  |


## Documentation
Check out the documentation and examples at https://trill.readthedocs.io/en/latest/index.html
