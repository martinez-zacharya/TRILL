import pytest
import esm
import pandas as pd
import os
import git
import torch
import pytorch_lightning as pl
import shutil
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from datasets import Dataset
from trill.utils.lightning_models import ProtGPT2
from trill.utils.update_weights import weights_update
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from trill.utils.protgpt2_utils import ProtGPT2_wrangle



@pytest.fixture
def get_git_root(path = os.getcwd()):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

@pytest.fixture
def fasta_import(get_git_root):
    data = esm.data.FastaBatchedDataset.from_file(os.path.join(get_git_root, 'trill/data/query.fasta'))
    yield data

def test_tokenizer_import():
    assert AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
    
@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
    yield tokenizer

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")    
def test_ProtGPT2_import():
    assert ProtGPT2(0.0001)

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
@pytest.fixture    
def ProtGPT2_import():
    model = ProtGPT2(0.0001)
    yield model
        
def test_fasta_wrangling(fasta_import, tokenizer):
    seq_dict_df = ProtGPT2_wrangle(fasta_import, tokenizer)
    assert len(seq_dict_df) == len(fasta_import)

@pytest.fixture    
def wrangle_fasta(fasta_import, tokenizer):
    yield ProtGPT2_wrangle(fasta_import, tokenizer)
    
@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_finetuning_ProtGPT2_gpu(wrangle_fasta, ProtGPT2_import):
    dataloader = torch.utils.data.DataLoader(wrangle_fasta, shuffle = False, batch_size = 1, num_workers=0)
    trainer = pl.Trainer(devices=2, profiler=None, accelerator='gpu', max_epochs=1, num_nodes = 1, precision = 16, amp_backend='native', strategy = 'deepspeed_stage_3')
    try:
        trainer.fit(model=ProtGPT2_import, train_dataloaders = dataloader)
    except Exception as exc:
        assert False, f"'test_finetuning_ProtGPT2_gpu()' raised an exception {exc}"

@pytest.fixture
@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def finetuned_ProtGPT2_gpu_fixture(wrangle_fasta, ProtGPT2_import, get_git_root):
    dataloader = torch.utils.data.DataLoader(wrangle_fasta, shuffle = False, batch_size = 1, num_workers=0)
    trainer = pl.Trainer(devices=2, profiler=None, accelerator='gpu', max_epochs=1, num_nodes = 1, precision = 16, amp_backend='native', strategy = 'deepspeed_stage_3')
    trainer.fit(model=ProtGPT2_import, train_dataloaders = dataloader)
    save_path = os.path.join(get_git_root, "lightning_logs/version_0/checkpoints/epoch=0-step=61.ckpt")
    output_path = "finetuned_protgpt2_pytest.pt"
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
    yield("finetuned_protgpt2_pytest.pt")
    shutil.rmtree(os.path.join(get_git_root, 'lightning_logs'))

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")   
def test_load_pretrained_ProtGPT2_gpu(finetuned_ProtGPT2_gpu_fixture):
    model = ProtGPT2()
    assert model.load_from_checkpoint(finetuned_ProtGPT2_gpu_fixture, strict=False)

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")       
def test_ProtGPT2_generate():
    model = ProtGPT2()
    generated_output = model.generate(seed_seq="M", max_length=25, do_sample = True, top_k=950, repetition_penalty=1.2, num_return_sequences=2)
    gen_seq_df = pd.DataFrame(generated_output, columns=['Generated_Sequence'])
    assert len(gen_seq_df) == 2