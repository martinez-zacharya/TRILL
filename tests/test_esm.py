import pytest
import esm
import sys
import torch
import git
import pandas as pd
import numpy as np
import os
import pytorch_lightning as pl
from esm.inverse_folding.util import load_structure, extract_coords_from_structure
from esm.inverse_folding.multichain_util import extract_coords_from_complex, sample_sequence_in_complex
from trill.utils.lightning_models import ESM
from trill.utils.update_weights import weights_update
from trill.utils.esm_utils import ESM_IF1_Wrangle, coordDataset, clean_embeddings, ESM_IF1


@pytest.fixture
def get_git_root(path = os.getcwd()):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root
        
def test_fasta_import(get_git_root):
    assert esm.data.FastaBatchedDataset.from_file(os.path.join(get_git_root, 'trill/data/query.fasta'))

@pytest.fixture
def fasta_import(get_git_root):
    data = esm.data.FastaBatchedDataset.from_file(os.path.join(get_git_root, 'trill/data/query.fasta'))
    yield data

def test_struct_import(get_git_root):
    assert ESM_IF1_Wrangle(os.path.join(get_git_root, 'trill/data/4ih9.pdb'))
    
@pytest.fixture
def struct_import(get_git_root):
    yield ESM_IF1_Wrangle(os.path.join(get_git_root, 'trill/data/4ih9.pdb'))

def test_ESM2_import():
    model_import_name = 'esm.pretrained.esm2_t6_8M_UR50D()'
    assert ESM(eval(model_import_name), 1e-5, False)

@pytest.fixture    
def ESM2_import():
    model_import_name = 'esm.pretrained.esm2_t6_8M_UR50D()'
    model = ESM(eval(model_import_name), 1e-5, False)
    yield model

# @pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
# def test_finetuning_ESM2_gpu(fasta_import, ESM2_import):
#     dataloader = torch.utils.data.DataLoader(fasta_import, shuffle = False, batch_size = 5, num_workers=0, collate_fn=ESM2_import.alphabet.get_batch_converter())
#     trainer = pl.Trainer(devices=1, profiler = None, accelerator='gpu', strategy = None, max_epochs=1, logger=None, num_nodes=1, precision = 16, amp_backend='native', enable_checkpointing=False)
#     try:
#         trainer.fit(model=ESM2_import, train_dataloaders=dataloader)
#     except Exception as exc:
#         assert False, f"'test_finetuning_ESM2_gpu()' raised an exception {exc}"

# @pytest.fixture
# @pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
# def finetuning_ESM2_gpu_fixture(fasta_import, ESM2_import):
#     dataloader = torch.utils.data.DataLoader(fasta_import, shuffle = False, batch_size = 5, num_workers=0, collate_fn=ESM2_import.alphabet.get_batch_converter())
#     trainer = pl.Trainer(devices=1, profiler = None, accelerator='gpu', strategy = None, max_epochs=1, logger=None, num_nodes=1, precision = 16, amp_backend='native', enable_checkpointing=False)
#     trainer.fit(model=ESM2_import, train_dataloaders=dataloader)
#     trainer.save_checkpoint(f"finetuned_esm2_pytest.pt")
#     yield("finetuned_esm2_pytest.pt")
#     os.remove("finetuned_esm2_pytest.pt")

# @pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
# def test_embed_base_ESM2_gpu(fasta_import, ESM2_import):
#     dataloader = torch.utils.data.DataLoader(fasta_import, shuffle = False, batch_size = 5, num_workers=0, collate_fn=ESM2_import.alphabet.get_batch_converter())
#     trainer = pl.Trainer(devices=1, profiler = None, accelerator='gpu', strategy = None, max_epochs=1, logger=None, num_nodes=1, precision = 16, amp_backend='native', enable_checkpointing=False)
#     trainer.predict(ESM2_import, dataloader)
#     finaldf = clean_embeddings(ESM2_import.reps)
#     assert len(finaldf) == len(fasta_import)

# @pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")   
# def test_load_pretrained_ESM2_gpu(finetuning_ESM2_gpu_fixture):
#     model_import_name = 'esm.pretrained.esm2_t6_8M_UR50D()'
#     assert weights_update(model = ESM(eval(model_import_name), 1e-5), checkpoint = torch.load(finetuning_ESM2_gpu_fixture))

# @pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")   
# def test_embed_pretrained_ESM2_gpu(fasta_import, finetuning_ESM2_gpu_fixture):
#     model_import_name = 'esm.pretrained.esm2_t6_8M_UR50D()'
#     model = weights_update(model = ESM(eval(model_import_name), 1e-5), checkpoint = torch.load(finetuning_ESM2_gpu_fixture))
#     dataloader = torch.utils.data.DataLoader(fasta_import, shuffle = False, batch_size = 5, num_workers=0, collate_fn=model.alphabet.get_batch_converter())
#     trainer = pl.Trainer(devices=1, profiler = None, accelerator='gpu', strategy = None, max_epochs=1, logger=None, num_nodes=1, precision = 16, amp_backend='native', enable_checkpointing=False)
#     trainer.predict(model, dataloader)
#     embeddings = pd.DataFrame(model.reps, columns = ['Embeddings', 'Label'])
#     finaldf = embeddings['Embeddings'].apply(pd.Series)
#     finaldf['Label'] = embeddings['Label']
#     assert len(finaldf) == len(fasta_import)
    
@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")   
def test_ESM2_IF1_gpu(struct_import):
    sample_df = ESM_IF1(struct_import, genIters=1, temp = 1.)
    assert len(sample_df) == 2
