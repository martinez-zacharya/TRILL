import pytest
import subprocess
import filecmp
import os
import git
import esm
import torch
from Bio import PDB
import pandas as pd
from trill.utils.lightning_models import ESM
import shutil

# Finetuning
###########################################################################################
@pytest.fixture    
@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def base_esm2_t12():
    model_import_name = 'esm.pretrained.esm2_t12_35M_UR50D()'
    model = ESM(eval(model_import_name), 1e-5, False)
    torch.save(model.state_dict(), "esm2_t12_35M_UR50D.pt")
    yield("esm2_t12_35M_UR50D.pt")
    os.remove("esm2_t12_35M_UR50D.pt")

@pytest.fixture
def get_git_root(path = os.getcwd()):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

@pytest.fixture
@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def finetune_esm2_t12():
    command = "trill test 1 finetune trill/data/query.fasta --epochs 2"
    command = command.split(" ")
    subprocess.run(command)
    yield("test_esm2_t12_35M_UR50D_2.pt")
    os.remove("test_esm2_t12_35M_UR50D_2.pt")

@pytest.fixture
@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def finetune_esm2_t12_rng456():
    command = "trill test_456 1 --RNG_seed 456 finetune trill/data/query.fasta --epochs 2"
    command = command.split(" ")
    subprocess.run(command)
    yield("test_456_esm2_t12_35M_UR50D_2.pt")
    os.remove("test_456_esm2_t12_35M_UR50D_2.pt")

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_finetune_esm2_t12(get_git_root, finetune_esm2_t12):
    assert filecmp.cmp("test_esm2_t12_35M_UR50D_2.pt", os.path.join(get_git_root, "trill/data/target_esm2_t12_35M_UR50D_2.pt"))

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_finetune_esm2_t12_456(get_git_root, finetune_esm2_t12_rng456):
    assert filecmp.cmp("test_456_esm2_t12_35M_UR50D_2.pt", os.path.join(get_git_root, "trill/data/target_456_esm2_t12_35M_UR50D_2.pt"))

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_rng_finetune_esm2_t12(get_git_root, finetune_esm2_t12, finetune_esm2_t12_rng456):
    assert not filecmp.cmp(os.path.join(get_git_root, "trill/data/target_esm2_t12_35M_UR50D_2.pt"), os.path.join(get_git_root, "trill/data/target_456_esm2_t12_35M_UR50D_2.pt"))

@pytest.fixture
@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def finetune_esm2_t30_deepspeed_stage_3_offload(get_git_root):
    command = "trill test_t30 1 finetune trill/data/query.fasta --model esm2_t30_150M_UR50D --epochs 2 --strategy deepspeed_stage_3_offload"
    command = command.split(" ")
    subprocess.run(command)
    yield("target_t30_esm2_2.pt.pt")
    shutil.rmtree(os.path.join(get_git_root, 'test_t30_esm2_t30_150M_UR50D_2.pt'))
    os.remove("test_t30_esm2_2.pt")

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_finetune_esm2_t30(get_git_root, finetune_esm2_t30_deepspeed_stage_3_offload):
    assert filecmp.cmp("test_t30_esm2_2.pt", os.path.join(get_git_root, "trill/data/target_t30_esm2_2.pt"))

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_finetuning_works(get_git_root, base_esm2_t12, finetune_esm2_t12):
    assert not filecmp.cmp("esm2_t12_35M_UR50D.pt", os.path.join(get_git_root, "test_esm2_t12_35M_UR50D_2.pt"))

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_finetune_bigbatch():
    command = 'trill test_bigbatch 1 finetune trill/data/query.fasta --epochs 2 --batch_size 2'
    command = command.split(" ")
    subprocess.run(command)
    os.remove('test_bigbatch_esm2_t12_35M_UR50D_2.pt')

# These next 2 need checking out!!!!!!!!!!!!!!!!!!!!!!!!
@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_finetune_protgpt2():
    command = 'trill test 1 finetune trill/data/query.fasta --model ProtGPT2 --epochs 2 --strategy deepspeed_stage_2_offload'
    command = command.split(" ")
    assert subprocess.run(command).check_returncode() == 0

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_finetune_protgpt2_bigbatch():
    command = 'trill test 1 finetune trill/data/query.fasta --model ProtGPT2 --epochs 2 --strategy deepspeed_stage_3_offload --batch_size 3'
    command = command.split(" ")
    assert subprocess.run(command).check_returncode() == 0

# Embedding
###########################################################################################
@pytest.fixture
@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def generate_base_embeddings():
    command = 'trill test 1 embed trill/data/query.fasta'
    command = command.split(" ")
    subprocess.run(command)
    yield('test_esm2_t12_35M_UR50D.csv')
    os.remove('test_esm2_t12_35M_UR50D.csv')

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_embed_base(get_git_root, generate_base_embeddings):
    assert filecmp.cmp("test_esm2_t12_35M_UR50D.csv", os.path.join(get_git_root, "trill/data/target_esm2_t12_35M_UR50D.csv"))

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_embeddings_bigbatch(get_git_root):
    command = 'trill test_bigbatch 1 embed trill/data/query.fasta --batch_size 3'
    command = command.split(" ")
    subprocess.run(command)
    assert len(pd.read_csv('test_bigbatch_esm2_t12_35M_UR50D.csv')) == len(pd.read_csv(os.path.join(get_git_root, 'trill/data/target_esm2_t12_35M_UR50D.csv')))
    os.remove('test_bigbatch_esm2_t12_35M_UR50D.csv')

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_embed_t30(get_git_root):
    command = 'trill test_t30 1 embed trill/data/query.fasta --model esm2_t30_150M_UR50D'
    command = command.split(" ")
    subprocess.run(command)
    assert len(pd.read_csv('test_t30_esm2_t30_150M_UR50D.csv')) == len(pd.read_csv(os.path.join(get_git_root, 'trill/data/target_esm2_t12_35M_UR50D.csv')))
    os.remove('test_t30_esm2_t30_150M_UR50D.csv')

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_embed_finetuned(get_git_root):
    command = 'trill test_tuned 1 embed trill/data/query.fasta --preTrained trill/data/target_esm2_t12_35M_UR50D_2.pt'
    command = command.split(" ")
    subprocess.run(command)
    assert not filecmp.cmp('test_tuned_esm2_t12_35M_UR50D.csv', os.path.join(get_git_root, 'trill/data/target_esm2_t12_35M_UR50D.csv'))
    os.remove('test_tuned_esm2_t12_35M_UR50D.csv')

# Generate
###########################################################################################
@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_if1(get_git_root):
    command = 'trill if1 1 inv_fold_gen ESM-IF1 --query trill/data/4ih9.pdb --genIters 3'
    command = command.split(" ")
    subprocess.run(command)
    assert filecmp.cmp('if1_IF1_gen.csv', os.path.join(get_git_root, 'trill/data/if1_target.csv'))
    os.remove('if1_IF1_gen.csv')

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_protgpt2_gen_base(get_git_root):
    command = 'trill gen_base 1 lang_gen ProtGPT2 --max_length 100 --num_return_sequences 5'
    command = command.split(" ")
    subprocess.run(command)
    seqkit_seq = 'seqkit seq -s gen_base_ProtGPT2.fasta'
    seqkit_seq = seqkit_seq.split(" ")
    seqkit_seq_out = subprocess.run(seqkit_seq, stdout=subprocess.PIPE).stdout.decode('ascii')
    seqkit_compare = f'seqkit seq -s trill/data/target_base_ProtGPT2.fasta'
    seqkit_compare = seqkit_compare.split(" ")
    seqkit_compare_out = subprocess.run(seqkit_compare, stdout=subprocess.PIPE).stdout.decode('ascii')
    assert seqkit_seq_out == seqkit_compare_out
    os.remove('gen_base_ProtGPT2.fasta')

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_protgpt2_gen_finetuned(get_git_root):
    command = 'trill gen_tuned 1 lang_gen ProtGPT2 --finetuned trill/data/I-D_ProtGPT2_10.pt --num_return_sequences 5 --max_length 100'
    command = command.split(" ")
    subprocess.run(command)
    seqkit_seq = 'seqkit seq -s gen_tuned_ProtGPT2.fasta'
    seqkit_seq = seqkit_seq.split(" ")
    seqkit_seq_out = subprocess.run(seqkit_seq, stdout=subprocess.PIPE).stdout.decode('ascii')
    seqkit_compare = f'seqkit seq -s trill/data/target_base_ProtGPT2.fasta'
    seqkit_compare = seqkit_compare.split(" ")
    seqkit_compare_out = subprocess.run(seqkit_compare, stdout=subprocess.PIPE).stdout.decode('ascii')
    assert seqkit_seq_out != seqkit_compare_out
    os.remove('gen_tuned_ProtGPT2.fasta')

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_mpnn(get_git_root):
    command = 'trill test 1 inv_fold_gen ProteinMPNN --query trill/data/4ih9.pdb --max_length 600 --num_return_sequences 3'
    command = command.split(" ")
    subprocess.run(command)
    seqkit_seq = 'seqkit seq -s ProteinMPNN_output/seqs/4ih9.fa'.split(" ")
    seqkit_seq_out = subprocess.run(seqkit_seq, stdout=subprocess.PIPE).stdout.decode('ascii')
    seqkit_compare = f'seqkit seq -s trill/data/target_4ih9.fa'.split(" ")
    seqkit_compare_out = subprocess.run(seqkit_compare, stdout=subprocess.PIPE).stdout.decode('ascii')
    assert seqkit_seq_out == seqkit_compare_out
    shutil.rmtree('ProteinMPNN_output')


@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_gibbs_base(get_git_root):
    command = 'trill test 1 lang_gen ESM2_Gibbs --esm2_arch esm2_t30_150M_UR50D --num_return_sequences 3'.split(" ")
    subprocess.run(command)
    seqkit_seq = 'seqkit seq -s test_esm2_t30_150M_UR50D_Gibbs.fasta'.split(" ")
    seqkit_seq_out = subprocess.run(seqkit_seq, stdout=subprocess.PIPE).stdout.decode('ascii')
    seqkit_compare = f'seqkit seq -s trill/data/target_esm2_t30_150M_UR50D_Gibbs.fasta'.split(" ")
    seqkit_compare_out = subprocess.run(seqkit_compare, stdout=subprocess.PIPE).stdout.decode('ascii')
    assert seqkit_seq_out == seqkit_compare_out
    os.remove('test_esm2_t30_150M_UR50D_Gibbs.fasta')

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_gibbs_tuned():
    command = 'trill test_tuned 1 lang_gen ESM2_Gibbs --esm2_arch esm2_t30_150M_UR50D --finetuned trill/data/target_esm2_t12_35M_UR50D_2.pt --num_return_sequences 3'.split(' ')
    subprocess.run(command)
    seqkit_seq = 'seqkit seq -s test_tuned_esm2_t30_150M_UR50D_Gibbs.fasta'.split(" ")
    seqkit_seq_out = subprocess.run(seqkit_seq, stdout=subprocess.PIPE).stdout.decode('ascii')
    seqkit_compare = f'seqkit seq -s trill/data/target_esm2_t30_150M_UR50D_Gibbs.fasta'.split(" ")
    seqkit_compare_out = subprocess.run(seqkit_compare, stdout=subprocess.PIPE).stdout.decode('ascii')
    assert seqkit_seq_out != seqkit_compare_out
    os.remove('test_tuned_esm2_t30_150M_UR50D_Gibbs.fasta')

@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_gibbs_t5():
    command = 'trill test_t5 1 lang_gen ESM2_Gibbs --esm2_arch esm2_t30_150M_UR50D --temp 5 --num_return_sequences 3'.split(" ")
    subprocess.run(command)
    seqkit_seq = 'seqkit seq -s test_t5_esm2_t30_150M_UR50D_Gibbs.fasta'.split(" ")
    seqkit_seq_out = subprocess.run(seqkit_seq, stdout=subprocess.PIPE).stdout.decode('ascii')
    seqkit_compare = f'seqkit seq -s trill/data/target_esm2_t30_150M_UR50D_Gibbs.fasta'.split(" ")
    seqkit_compare_out = subprocess.run(seqkit_compare, stdout=subprocess.PIPE).stdout.decode('ascii')
    assert seqkit_seq_out != seqkit_compare_out
    os.remove('test_t5_esm2_t30_150M_UR50D_Gibbs.fasta')

# Fold
###########################################################################################
@pytest.mark.skipif(torch.cuda.is_available() == False, reason = "GPU is not available")
def test_esmfold():
    command = 'trill cpp_ex 1 fold trill/data/cpp_ex.fasta'.split(' ')
    subprocess.run(command)
    parser = PDB.PDBParser()
    test = list(parser.get_structure("test", "pos_518.pdb").get_atoms())
    target = list(parser.get_structure("target", "trill/data/target_cpp_ex.pdb").get_atoms())
    assert test == target
    os.remove('pos_518.pdb')

# Visualize
###########################################################################################
def test_viz():
    pca = 'trill viz 1 visualize trill/data/target_esm2_t12_35M_UR50D.csv'.split(' ')
    subprocess.run(pca)
    pca2 = 'trill viz2 1 visualize trill/data/target_esm2_t12_35M_UR50D.csv'.split(' ')
    subprocess.run(pca2)
    umap = 'trill viz 1 visualize trill/data/target_esm2_t12_35M_UR50D.csv --method UMAP'.split(' ')
    subprocess.run(umap)
    tsne = 'trill viz 1 visualize trill/data/target_esm2_t12_35M_UR50D.csv --method tSNE'.split(' ')
    subprocess.run(tsne)
    assert not pd.read_csv('viz_PCA_target_esm2_t12_35M_UR50D.csv').equals(pd.read_csv('viz_UMAP_target_esm2_t12_35M_UR50D.csv'))
    assert not pd.read_csv('viz_PCA_target_esm2_t12_35M_UR50D.csv').equals(pd.read_csv('viz_tSNE_target_esm2_t12_35M_UR50D.csv'))
    assert not pd.read_csv('viz_tSNE_target_esm2_t12_35M_UR50D.csv').equals(pd.read_csv('viz_UMAP_target_esm2_t12_35M_UR50D.csv'))
    assert pd.read_csv('viz_PCA_target_esm2_t12_35M_UR50D.csv').equals(pd.read_csv('viz2_PCA_target_esm2_t12_35M_UR50D.csv'))
    os.remove('viz_PCA_target_esm2_t12_35M_UR50D.csv')
    os.remove('viz_PCA_target_esm2_t12_35M_UR50D.html')
    os.remove('viz2_PCA_target_esm2_t12_35M_UR50D.csv')
    os.remove('viz2_PCA_target_esm2_t12_35M_UR50D.html')
    os.remove('viz_tSNE_target_esm2_t12_35M_UR50D.csv')
    os.remove('viz_tSNE_target_esm2_t12_35M_UR50D.html')
    os.remove('viz_UMAP_target_esm2_t12_35M_UR50D.csv')
    os.remove('viz_UMAP_target_esm2_t12_35M_UR50D.html')

# Diffusion