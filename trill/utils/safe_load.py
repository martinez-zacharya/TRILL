"""
Utility functions for safely loading PyTorch models with safetensors support.
"""
import os
import torch
import warnings
from typing import Dict, Any, Optional, Union
from pathlib import Path

try:
    from safetensors.torch import load_file as safetensors_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    warnings.warn("safetensors not available. Consider installing it for safer model loading.")


def safe_torch_load(
    file_path: Union[str, Path], 
    map_location: Optional[Union[str, torch.device]] = None,
    weights_only: bool = True,
    prefer_safetensors: bool = True
) -> Dict[str, Any]:
    """
    Safely load a PyTorch model, with safetensors support when available.
    
    Args:
        file_path: Path to the model file (.pt, .pth, or .safetensors)
        map_location: Device to map tensors to
        weights_only: Only load weights (safer for .pt files)
        prefer_safetensors: Try to load .safetensors version first if available
        
    Returns:
        Dictionary containing the loaded model state
        
    Raises:
        FileNotFoundError: If neither .pt nor .safetensors file exists
        ValueError: If PyTorch version is too old for safe loading
    """
    file_path = Path(file_path)
    
    # If explicit safetensors file is provided
    if file_path.suffix == '.safetensors':
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors is required to load .safetensors files")
        device_str = str(map_location) if map_location is not None else "cpu"
        return {"state_dict": safetensors_load_file(str(file_path), device=device_str)}
    
    # For .pt/.pth files, check if safetensors alternative exists
    if prefer_safetensors and SAFETENSORS_AVAILABLE:
        safetensors_path = file_path.with_suffix('.safetensors')
        if safetensors_path.exists():
            print(f"Found safetensors version at {safetensors_path}, loading for safety")
            device_str = str(map_location) if map_location is not None else "cpu"
            return {"state_dict": safetensors_load_file(str(safetensors_path), device=device_str)}
    
    # Fall back to torch.load with safety checks
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    # Check PyTorch version for security
    torch_version = torch.__version__
    major, minor = map(int, torch_version.split('.')[:2])
    
    if major < 2 or (major == 2 and minor < 6):
        warnings.warn(
            f"PyTorch version {torch_version} has security vulnerabilities in torch.load. "
            f"Consider upgrading to PyTorch 2.6+ or use safetensors format.",
            UserWarning
        )
    
    try:
        return torch.load(file_path, map_location=map_location, weights_only=weights_only)
    except Exception as e:
        if "weights_only=True" in str(e):
            # Fallback for older models that need weights_only=False
            warnings.warn(
                "Loading with weights_only=False due to compatibility issues. "
                "Consider converting to safetensors format for better security.",
                UserWarning
            )
            return torch.load(file_path, map_location=map_location, weights_only=False)
        raise


def safe_torch_save(
    obj: Dict[str, Any], 
    file_path: Union[str, Path],
    save_safetensors: bool = True,
    save_pytorch: bool = True
) -> None:
    """
    Save a PyTorch model in both safetensors and PyTorch formats.
    
    Args:
        obj: Dictionary containing model state to save
        file_path: Path to save the model (without extension)
        save_safetensors: Whether to save in safetensors format
        save_pytorch: Whether to save in PyTorch format
    """
    file_path = Path(file_path)
    
    # Remove extension if provided
    if file_path.suffix in ['.pt', '.pth', '.safetensors']:
        file_path = file_path.with_suffix('')
    
    if save_pytorch:
        pt_path = file_path.with_suffix('.pt')
        torch.save(obj, pt_path)
        print(f"Saved PyTorch model to {pt_path}")
    
    if save_safetensors and SAFETENSORS_AVAILABLE:
        from safetensors.torch import save_file as safetensors_save_file
        safetensors_path = file_path.with_suffix('.safetensors')
        
        # Extract state_dict if it's wrapped in a dictionary
        if isinstance(obj, dict) and 'state_dict' in obj:
            state_dict = obj['state_dict']
        elif isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            state_dict = obj
        else:
            warnings.warn("Cannot extract state_dict for safetensors saving. Skipping safetensors format.")
            return
            
        safetensors_save_file(state_dict, str(safetensors_path))
        print(f"Saved safetensors model to {safetensors_path}")


def convert_pt_to_safetensors(pt_path: Union[str, Path], safetensors_path: Optional[Union[str, Path]] = None) -> None:
    """
    Convert a PyTorch .pt file to safetensors format.
    
    Args:
        pt_path: Path to the .pt file
        safetensors_path: Output path for safetensors file (optional, defaults to same name with .safetensors extension)
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required for conversion")
    
    pt_path = Path(pt_path)
    if safetensors_path is None:
        safetensors_path = pt_path.with_suffix('.safetensors')
    else:
        safetensors_path = Path(safetensors_path)
    
    # Load the PyTorch model
    print(f"Loading PyTorch model from {pt_path}")
    try:
        checkpoint = torch.load(pt_path, map_location='cpu', weights_only=True)
    except:
        checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
        warnings.warn("Loaded with weights_only=False - model may contain non-tensor data")
    
    # Extract state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        raise ValueError("Cannot extract state_dict from checkpoint")
    
    # Save as safetensors
    from safetensors.torch import save_file as safetensors_save_file
    safetensors_save_file(state_dict, str(safetensors_path))
    print(f"Converted to safetensors format: {safetensors_path}")
