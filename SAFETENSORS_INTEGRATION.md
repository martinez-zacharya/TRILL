# Safetensors Integration Summary

## Changes Made to TRILL for Safetensors Support

### 1. Created Safe Loading Utility (`trill/utils/safe_load.py`)
- `safe_torch_load()`: Safely loads models with safetensors preference
- `safe_torch_save()`: Saves models in both PyTorch and safetensors formats
- `convert_pt_to_safetensors()`: Converts existing .pt files to safetensors
- Automatic fallback to torch.load with security warnings
- Support for PyTorch 2.6+ security requirements

### 2. Updated Core Commands
#### Finetune (`trill/commands/finetune.py`)
- Added safetensors import
- Replaced `torch.load()` calls with `safe_torch_load()`
- Support for safer fine-tuned model loading

#### Embed (`trill/commands/embed.py`)
- Added safetensors import
- Replaced `torch.load()` calls with `safe_torch_load()`
- Safer loading of pre-trained embeddings

#### Language Generation (`trill/commands/lang_gen.py`)
- Added safetensors import
- Replaced `torch.load()` calls with `safe_torch_load()`
- Safer loading of fine-tuned language models

#### Classify (`trill/commands/classify.py`)
- Added safetensors import
- Replaced multiple `torch.load()` calls with `safe_torch_load()`
- Safer loading of pre-computed embeddings and model checkpoints

#### Fold (`trill/commands/fold.py`)
- Added safetensors import
- Replaced `torch.load()` calls with `safe_torch_load()`
- Safer loading of fold predictions

#### Dock (`trill/commands/dock.py`)
- Added safetensors import
- Replaced `torch.load()` calls with `safe_torch_load()`
- Safer loading of docking embeddings

### 3. Updated Utility Files
#### Classification Utils (`trill/utils/classify_utils.py`)
- Added safetensors import
- Replaced `torch.load()` calls with `safe_torch_load()`

#### Inverse Folding Utils (`trill/utils/inverse_folding/util.py`)
- Added safetensors import
- Replaced `torch.load()` calls with `safe_torch_load()`

#### EpHod Utils (`trill/utils/ephod_utils.py`)
- Added safetensors import
- Replaced `torch.load()` calls with `safe_torch_load()`

#### PSICHIC Utils (`trill/utils/psichic.py`)
- Added safetensors import
- Replaced `torch.load()` calls with `safe_torch_load()`

#### M-Ionic Utils (`trill/utils/mionic_utils.py`)
- Added safetensors import
- Replaced `torch.load()` calls with `safe_torch_load()`

#### Lightning Models (`trill/utils/lightning_models.py`)
- Added safetensors import
- Replaced `torch.load()` calls with `safe_torch_load()`

#### Pool PaRTI Utils (`trill/utils/poolparti.py`)
- Added safetensors import
- Replaced `torch.load()` calls with `safe_torch_load()`

### 4. Added Conversion Utility
#### Utils Command (`trill/commands/utils.py`)
- Added `convert_to_safetensors` tool option
- Users can now convert .pt files to safetensors via:
  ```bash
  trill 0 utils convert_to_safetensors --pt_file model.pt
  ```

### 5. Updated Documentation (`docs/home.md`)
- Added comprehensive safetensors section in "Misc. Tips"
- Explained security benefits and automatic detection
- Provided examples for manual conversion
- Referenced CVE-2025-32434 security vulnerability

## Benefits

1. **Security**: Eliminates code execution vulnerabilities in torch.load
2. **Performance**: Faster loading with memory mapping
3. **Backward Compatibility**: Automatic fallback to .pt files when needed
4. **User-Friendly**: Automatic detection and conversion utilities
5. **Future-Proof**: Aligns with modern ML security best practices

## Usage Examples

### Automatic Safetensors Detection
```bash
# TRILL will automatically look for model.safetensors first, then model.pt
trill 1 finetune esm2_t12_35M query.fasta --finetuned model.pt
```

### Manual Conversion
```bash
# Convert existing .pt file to safetensors
trill 0 utils convert_to_safetensors --pt_file finetuned_model.pt
```

### Python API
```python
from trill.utils.safe_load import safe_torch_load, convert_pt_to_safetensors

# Safe loading with automatic safetensors preference
checkpoint = safe_torch_load("model.pt")

# Manual conversion
convert_pt_to_safetensors("model.pt", "model.safetensors")
```

## Security Impact

This update addresses CVE-2025-32434, a serious vulnerability in PyTorch's `torch.load()` function that allows arbitrary code execution. The safetensors format is immune to this vulnerability as it cannot execute arbitrary code during loading.
