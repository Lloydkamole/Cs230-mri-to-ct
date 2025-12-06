"""
PFGM-FIR Inference Script for Full CT Volume Generation.

Generates complete CT volumes from MRI NIfTI files using trained PFGM-FIR models.

Usage:
    # Option 1: Edit the config below and run directly:
    python inference.py
    
    # Option 2: Use command line arguments:
    python inference.py --checkpoint path/to/checkpoint --input mri.nii.gz --output pred_ct.nii.gz
    
    # Entire folder of MRI files:
    python inference.py --checkpoint path/to/checkpoint --input_dir /path/to/mris --output_dir /path/to/outputs
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nibabel as nib

# MONAI transforms for consistent data loading (matching training)
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
)

# Import model
from models import create_swin_fir_unet

# For loading Accelerate checkpoints
from safetensors.torch import load_file as load_safetensors


# =============================================================================
# *** USER CONFIGURATION - EDIT THESE VALUES ***
# =============================================================================

# Paths - Set these to run directly with `python inference.py`
INFERENCE_CONFIG = {
    # ---------- INPUT/OUTPUT PATHS ----------
    # For single file inference:
    "input_mri": "/home/lloyd/Documents/AR PFGM/test_pfgm/1PA180/mr.nii_processed_patch.nii.gz",           # e.g., "/path/to/patient/mr.nii.gz"
    "output_ct": "/home/lloyd/Documents/AR PFGM/test_pfgm/1PA180/pred_ct.nii.gz",           # e.g., "/path/to/patient/pred_ct.nii.gz" (auto-generated if None)
    
    # For batch/directory inference (overrides single file if set):
    "input_dir": None,           # e.g., "/path/to/patients_folder"
    "output_dir": None,          # e.g., "/path/to/output_folder" (auto-generated if None)
    "mr_pattern": "mr.nii",      # Pattern to match MRI files in directory mode
    
    # ---------- CHECKPOINT PATH ----------
    "checkpoint": "/home/lloyd/Documents/MDLM_unet/MDLM_UNET/pfgm_fir/outputs_accelerate/20251204_031357_pfgm_fir/checkpoints/best",
    
    # ---------- SAMPLING PARAMETERS ----------
    "num_steps": 100,             # Number of sampling steps (more = better quality, slower)
    "cfg_scale": 2.0,            # Classifier-free guidance scale (1.0 = no guidance, try 1.5-3.0)
    "batch_size": 12,             # Batch size for slice processing
    "target_size": 256,          # Target size for 2D slices
    
    # ---------- SYSTEM ----------
    "device": "cuda:0",            # "cuda", "cuda:0", "cuda:1", or "cpu"
    "seed": 42,                  # Random seed for reproducibility
    "num_workers": 4,            # Number of data loading workers (0 = main process only)
    "prefetch_factor": 2,        # Number of batches to prefetch per worker
}

# =============================================================================
# Model Architecture Configuration (must match training)
# =============================================================================

DEFAULT_MODEL_CONFIG = {
    # Model Architecture - MUST MATCH TRAINING CONFIG (train_accelerate.py)
    "base_channels": 64,
    "channel_mults": [1, 2, 4, 8],  # From train_accelerate.py
    "num_res_blocks": 3,
    "attention_resolutions": [2, 4, 8],  # From train_accelerate.py
    "num_heads": 4,
    "mlp_ratio": 4.0,
    "radius_emb_dim": 256,
    "window_size": 4,
    "use_sfb": True,
    "dropout": 0.0,  # Match training
    "drop_path": 0.1,
    
    # EDM Parameters
    "sigma_min": 0.002,
    "sigma_max": 80.0,
    "sigma_data": 0.5,
    "rho": 7.0,
}

DEFAULT_SAMPLING_CONFIG = {
    "num_steps": 50,
    "S_churn": 0.0,
    "S_min": 0.0,
    "S_max": float('inf'),
    "S_noise": 1.0,
    "cfg_scale": 1.0,
}


# =============================================================================
# EDM Sampling Functions
# =============================================================================

def edm_precond(
    model: torch.nn.Module,
    x: torch.Tensor,
    mri: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """Apply EDM preconditioning to model output."""
    x = x.float()
    mri = mri.float()
    sigma = sigma.float()
    
    sigma_4d = sigma.view(-1, 1, 1, 1)
    c_skip = sigma_data ** 2 / (sigma_4d ** 2 + sigma_data ** 2)
    c_out = sigma_4d * sigma_data / (sigma_4d ** 2 + sigma_data ** 2).sqrt()
    c_in = 1.0 / (sigma_4d ** 2 + sigma_data ** 2).sqrt()
    c_noise = sigma.log() / 4
    
    x_scaled = c_in * x
    F_x = model(x_scaled, mri, c_noise)
    D_x = c_skip * x + c_out * F_x
    
    return D_x


@torch.no_grad()
def edm_sample(
    model: torch.nn.Module,
    mri: torch.Tensor,
    num_steps: int = 50,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    rho: float = 7.0,
    S_churn: float = 0.0,
    S_min: float = 0.0,
    S_max: float = float('inf'),
    S_noise: float = 1.0,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """
    Generate CT images from MRI using EDM sampling (Algorithm 2 from EDM paper).
    
    Supports classifier-free guidance when cfg_scale > 1.0.
    
    Args:
        model: The trained denoising model
        mri: Input MRI tensor [B, 1, H, W]
        num_steps: Number of sampling steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        sigma_data: Data standard deviation (EDM parameter)
        rho: Schedule parameter
        S_churn: Stochasticity amount
        S_min: Minimum sigma for stochasticity
        S_max: Maximum sigma for stochasticity
        S_noise: Noise scaling for stochasticity
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
    
    Returns:
        Generated CT tensor [B, 1, H, W]
    """
    B, C, H, W = mri.shape
    device = mri.device
    
    # Time step discretization
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    
    # Initial noise
    x_next = torch.randn(B, 1, H, W, device=device, dtype=torch.float64) * t_steps[0]
    
    # Unconditional input for CFG (zeros)
    mri_uncond = torch.zeros_like(mri) if cfg_scale != 1.0 else None
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # Increase noise temporarily (stochasticity)
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        
        sigma_batch = torch.full((B,), t_hat.item(), device=device, dtype=torch.float32)
        
        # Denoising step with optional CFG
        if cfg_scale != 1.0 and mri_uncond is not None:
            # Conditional prediction
            denoised_cond = edm_precond(model, x_hat.float(), mri, sigma_batch, sigma_data).double()
            # Unconditional prediction
            denoised_uncond = edm_precond(model, x_hat.float(), mri_uncond, sigma_batch, sigma_data).double()
            # CFG combination
            denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
        else:
            denoised = edm_precond(model, x_hat.float(), mri, sigma_batch, sigma_data).double()
        
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        
        # 2nd order correction (Heun's method)
        if i < num_steps - 1 and t_next > 0:
            sigma_batch = torch.full((B,), t_next.item(), device=device, dtype=torch.float32)
            
            if cfg_scale != 1.0 and mri_uncond is not None:
                denoised_cond = edm_precond(model, x_next.float(), mri, sigma_batch, sigma_data).double()
                denoised_uncond = edm_precond(model, x_next.float(), mri_uncond, sigma_batch, sigma_data).double()
                denoised_next = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
            else:
                denoised_next = edm_precond(model, x_next.float(), mri, sigma_batch, sigma_data).double()
            
            d_prime = (x_next - denoised_next) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    
    return x_next.float()


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    checkpoint_path: str,
    model_config: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Load a trained PFGM-FIR model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory or model file
        model_config: Model configuration dict (uses defaults if None)
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Use default config if not provided
    cfg = model_config or DEFAULT_MODEL_CONFIG.copy()
    
    # Create model
    model = create_swin_fir_unet(
        config="custom",
        in_channels=2,
        out_channels=1,
        base_channels=cfg["base_channels"],
        channel_mults=cfg["channel_mults"],
        num_res_blocks=cfg["num_res_blocks"],
        attention_resolutions=cfg["attention_resolutions"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        radius_emb_dim=cfg["radius_emb_dim"],
        window_size=cfg["window_size"],
        use_sfb=cfg["use_sfb"],
        dropout=cfg["dropout"],
        drop_path_rate=cfg["drop_path"],
    )
    
    # Determine checkpoint file path
    if checkpoint_path.is_dir():
        # Accelerate checkpoint directory
        safetensors_path = checkpoint_path / "model.safetensors"
        pytorch_path = checkpoint_path / "pytorch_model.bin"
        
        if safetensors_path.exists():
            print(f"Loading from safetensors: {safetensors_path}")
            state_dict = load_safetensors(str(safetensors_path))
        elif pytorch_path.exists():
            print(f"Loading from PyTorch bin: {pytorch_path}")
            state_dict = torch.load(str(pytorch_path), map_location="cpu")
        else:
            raise FileNotFoundError(f"No model file found in {checkpoint_path}")
    else:
        # Direct model file
        if checkpoint_path.suffix == ".safetensors":
            state_dict = load_safetensors(str(checkpoint_path))
        else:
            state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    
    # Load state dict (handle potential "module." prefix from DDP)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params/1e6:.2f}M parameters")
    
    return model


# =============================================================================
# Volume Processing
# =============================================================================

def load_mri_volume(
    mri_path: str,
    target_size: int = 256,
) -> tuple:
    """
    Load MRI NIfTI volume and prepare for inference.
    
    Uses the SAME MONAI transforms as training (loader.py) to ensure consistent preprocessing:
    - LoadImaged: Load NIfTI file
    - EnsureChannelFirstd: Add channel dimension
    - Orientationd: Reorient to RAS standard orientation
    
    Args:
        mri_path: Path to MRI NIfTI file
        target_size: Target size for 2D slices
    
    Returns:
        Tuple of (preprocessed_slices, original_nifti, original_shape)
    """
    print(f"Loading MRI: {mri_path}")
    
    # Keep original NIfTI for saving output with same affine/header
    original_nii = nib.load(mri_path)
    original_shape = original_nii.shape
    
    # Use MONAI transforms EXACTLY matching training (loader.py)
    # This ensures consistent preprocessing: orientation, channel order, etc.
    inference_transforms = Compose([
        LoadImaged(keys=["mr"], image_only=True),
        EnsureChannelFirstd(keys=["mr"]),
        Orientationd(keys=["mr"], axcodes="RAS"),  # Critical: match training orientation
    ])
    
    # Apply transforms
    data = inference_transforms({"mr": mri_path})
    mri_vol = data["mr"]  # [C, H, W, D] tensor
    
    # Convert to numpy if needed
    if isinstance(mri_vol, torch.Tensor):
        mri_data = mri_vol.numpy()
    else:
        mri_data = np.array(mri_vol)
    
    print(f"  Original shape: {original_shape}")
    print(f"  After MONAI transforms: {mri_data.shape} (C, H, W, D)")
    print(f"  Value range: [{mri_data.min():.3f}, {mri_data.max():.3f}]")
    
    # Process slices from the [C, H, W, D] volume
    # C=0 since we have single channel
    num_slices = mri_data.shape[-1]
    slices = []
    
    for s in range(num_slices):
        # Extract slice: [C, H, W, D] -> [C, H, W]
        slice_2d = mri_data[:, :, :, s]  # [C, H, W]
        
        # Convert to tensor
        slice_tensor = torch.from_numpy(slice_2d).float()  # [C, H, W]
        
        # Resize if needed (matching training loader)
        h, w = slice_tensor.shape[1], slice_tensor.shape[2]
        if h != target_size or w != target_size:
            slice_tensor = F.interpolate(
                slice_tensor.unsqueeze(0),  # [1, C, H, W]
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # [C, H, W]
        
        slices.append(slice_tensor)
    
    print(f"  Prepared {num_slices} slices at {target_size}x{target_size}")
    
    return slices, original_nii, original_shape


class SliceDataset(Dataset):
    """Simple dataset wrapper for MRI slices to enable DataLoader with workers."""
    
    def __init__(self, slices: list):
        self.slices = slices
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        return idx, self.slices[idx]


def generate_ct_volume(
    model: torch.nn.Module,
    mri_slices: list,
    batch_size: int = 4,
    num_steps: int = 50,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    rho: float = 7.0,
    cfg_scale: float = 1.0,
    device: str = "cuda",
    show_progress: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> list:
    """
    Generate CT slices from MRI slices using the trained model.
    
    Args:
        model: Trained PFGM-FIR model
        mri_slices: List of MRI slice tensors [1, H, W]
        batch_size: Batch size for inference
        num_steps: Number of sampling steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        sigma_data: Data standard deviation
        rho: Schedule parameter
        cfg_scale: Classifier-free guidance scale
        device: Device for inference
        show_progress: Whether to show progress bar
        num_workers: Number of data loading workers
        prefetch_factor: Number of batches to prefetch per worker
    
    Returns:
        List of generated CT slice tensors (in correct order)
    """
    model.eval()
    num_slices = len(mri_slices)
    
    # Create dataset and dataloader with multiple workers
    dataset = SliceDataset(mri_slices)
    
    # Note: num_workers=0 means single-process (no multiprocessing)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,  # Must be False to maintain slice order
        "num_workers": num_workers,
        "pin_memory": True if device.startswith("cuda") else False,
        "drop_last": False,
    }
    
    # prefetch_factor only valid when num_workers > 0
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True
    
    dataloader = DataLoader(dataset, **loader_kwargs)
    
    # Store results with indices to maintain order
    results = {}
    
    pbar = tqdm(dataloader, desc="Generating CT", disable=not show_progress, total=len(dataloader))
    
    for batch_indices, batch_mri in pbar:
        # Move to device
        batch_mri = batch_mri.to(device)  # [B, 1, H, W]
        
        # Generate CT
        with torch.no_grad():
            batch_ct = edm_sample(
                model, batch_mri,
                num_steps=num_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sigma_data=sigma_data,
                rho=rho,
                cfg_scale=cfg_scale,
            )
        
        # Clamp to valid range (model outputs in [-1, 1] range)
        batch_ct = torch.clamp(batch_ct, -1.0, 1.0)
        
        # Store individual slices with their indices
        for i, idx in enumerate(batch_indices.tolist()):
            results[idx] = batch_ct[i].cpu()
        
        pbar.set_postfix({"slices": f"{len(results)}/{num_slices}"})
    
    # Reconstruct ordered list
    ct_slices = [results[i] for i in range(num_slices)]
    
    return ct_slices


def save_ct_volume(
    ct_slices: list,
    reference_nii: nib.Nifti1Image,
    original_shape: tuple,
    output_path: str,
    target_size: int = 256,
) -> None:
    """
    Save generated CT slices as a NIfTI volume.
    
    Args:
        ct_slices: List of CT slice tensors [1, H, W]
        reference_nii: Reference NIfTI for affine and header
        original_shape: Original volume shape for resizing back
        output_path: Output file path
        target_size: Size used during inference
    """
    print(f"Saving CT volume: {output_path}")
    
    # Stack slices into volume
    ct_volume = torch.stack(ct_slices).squeeze(1).numpy()  # [D, H, W]
    ct_volume = ct_volume.transpose(1, 2, 0)  # [H, W, D]
    
    print(f"  Generated shape: {ct_volume.shape}")
    
    # Resize back to original spatial dimensions if different
    orig_h, orig_w = original_shape[0], original_shape[1]
    if ct_volume.shape[0] != orig_h or ct_volume.shape[1] != orig_w:
        print(f"  Resizing from ({ct_volume.shape[0]}, {ct_volume.shape[1]}) to ({orig_h}, {orig_w})")
        
        # Convert to tensor for interpolation
        ct_tensor = torch.from_numpy(ct_volume).float()
        ct_tensor = ct_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]
        
        # Resize each slice
        resized_slices = []
        for s in range(ct_tensor.shape[1]):
            slice_resized = F.interpolate(
                ct_tensor[:, s:s+1, :, :],
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            )
            resized_slices.append(slice_resized.squeeze(0))
        
        ct_tensor = torch.stack(resized_slices, dim=0).squeeze(1)  # [D, H, W]
        ct_volume = ct_tensor.numpy().transpose(1, 2, 0)  # [H, W, D]
    
    # Keep output values as-is from the model (no conversion)
    
    print(f"  Final shape: {ct_volume.shape}")
    print(f"  Value range: [{ct_volume.min():.3f}, {ct_volume.max():.3f}]")
    
    # Create NIfTI image with same affine/header as input
    ct_nii = nib.Nifti1Image(ct_volume.astype(np.float32), reference_nii.affine, reference_nii.header)
    
    # Save
    nib.save(ct_nii, output_path)
    print(f"  Saved: {output_path}")


# =============================================================================
# Main Inference Functions
# =============================================================================

def infer_single(
    model: torch.nn.Module,
    mri_path: str,
    output_path: str,
    target_size: int = 256,
    batch_size: int = 4,
    num_steps: int = 50,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    rho: float = 7.0,
    cfg_scale: float = 1.0,
    device: str = "cuda",
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> None:
    """
    Generate CT volume from a single MRI file.
    
    Args:
        model: Trained model
        mri_path: Path to input MRI NIfTI
        output_path: Path for output CT NIfTI
        target_size: Target size for processing
        batch_size: Batch size for inference
        num_steps: Number of sampling steps
        sigma_min/sigma_max/sigma_data/rho: EDM parameters
        cfg_scale: Classifier-free guidance scale
        device: Device for inference
        num_workers: Number of data loading workers
        prefetch_factor: Number of batches to prefetch per worker
    """
    # Load MRI
    mri_slices, reference_nii, original_shape = load_mri_volume(mri_path, target_size)
    
    # Generate CT
    ct_slices = generate_ct_volume(
        model, mri_slices,
        batch_size=batch_size,
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
        rho=rho,
        cfg_scale=cfg_scale,
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    
    # Save CT
    save_ct_volume(ct_slices, reference_nii, original_shape, output_path, target_size)


def infer_directory(
    model: torch.nn.Module,
    input_dir: str,
    output_dir: str,
    mr_pattern: str = "mr.nii",
    target_size: int = 256,
    batch_size: int = 4,
    num_steps: int = 50,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    rho: float = 7.0,
    cfg_scale: float = 1.0,
    device: str = "cuda",
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> None:
    """
    Generate CT volumes for all MRI files in a directory.
    
    Args:
        model: Trained model
        input_dir: Directory containing MRI files
        output_dir: Directory for output CT files
        mr_pattern: Pattern to match MRI files
        Other args: Same as infer_single
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find MRI files
    mri_files = []
    
    # Check if input_dir contains patient subdirectories
    for item in sorted(input_dir.iterdir()):
        if item.is_dir():
            # Look for MRI in subdirectory
            for f in item.iterdir():
                if mr_pattern in f.name:
                    mri_files.append((f, item.name))
                    break
        elif item.is_file() and (mr_pattern in item.name or item.suffix in ['.nii', '.gz']):
            mri_files.append((item, item.stem))
    
    if not mri_files:
        print(f"No MRI files found matching '{mr_pattern}' in {input_dir}")
        return
    
    print(f"\nFound {len(mri_files)} MRI files to process\n")
    
    for mri_path, patient_id in mri_files:
        print(f"\n{'='*60}")
        print(f"Processing: {patient_id}")
        print(f"{'='*60}")
        
        output_path = output_dir / f"{patient_id}_pred_ct.nii.gz"
        
        try:
            infer_single(
                model, str(mri_path), str(output_path),
                target_size=target_size,
                batch_size=batch_size,
                num_steps=num_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sigma_data=sigma_data,
                rho=rho,
                cfg_scale=cfg_scale,
                device=device,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Complete! Results saved to: {output_dir}")
    print(f"{'='*60}\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="PFGM-FIR Inference: Generate CT volumes from MRI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument(
        "--checkpoint", "-c",
        type=str, required=True,
        help="Path to checkpoint directory or model file"
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i",
        type=str, default=None,
        help="Path to single input MRI NIfTI file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Path for single output CT NIfTI file"
    )
    parser.add_argument(
        "--input_dir",
        type=str, default=None,
        help="Directory containing input MRI files"
    )
    parser.add_argument(
        "--output_dir",
        type=str, default=None,
        help="Directory for output CT files"
    )
    parser.add_argument(
        "--mr_pattern",
        type=str, default="mr.nii",
        help="Pattern to match MRI files in directory mode"
    )
    
    # Model config
    parser.add_argument(
        "--base_channels", type=int, default=64,
        help="Model base channels"
    )
    parser.add_argument(
        "--channel_mults", type=str, default="1,2,4",
        help="Channel multipliers (comma-separated) - must match training config"
    )
    
    # Sampling parameters
    parser.add_argument(
        "--num_steps", "-n",
        type=int, default=50,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float, default=1.0,
        help="Classifier-free guidance scale (1.0 = no guidance)"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int, default=4,
        help="Batch size for slice processing"
    )
    parser.add_argument(
        "--target_size",
        type=int, default=256,
        help="Target size for 2D slices"
    )
    
    # EDM parameters
    parser.add_argument("--sigma_min", type=float, default=0.002, help="Minimum sigma")
    parser.add_argument("--sigma_max", type=float, default=80.0, help="Maximum sigma")
    parser.add_argument("--sigma_data", type=float, default=0.5, help="Data sigma")
    parser.add_argument("--rho", type=float, default=7.0, help="Schedule rho")
    
    # System
    parser.add_argument(
        "--device",
        type=str, default="cuda",
        help="Device for inference (cuda, cpu, cuda:0, etc.)"
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_workers", "-w",
        type=int, default=4,
        help="Number of data loading workers (0 = single process)"
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int, default=2,
        help="Number of batches to prefetch per worker"
    )
    
    return parser.parse_args()


def run_from_config():
    """
    Run inference using the INFERENCE_CONFIG defined at the top of the file.
    This is called when no command line arguments are provided.
    """
    cfg = INFERENCE_CONFIG
    
    # Validate config
    has_single = cfg["input_mri"] is not None
    has_batch = cfg["input_dir"] is not None
    
    if not has_single and not has_batch:
        print("="*60)
        print("ERROR: No input specified!")
        print("="*60)
        print("\nPlease edit INFERENCE_CONFIG at the top of this file:")
        print('  - Set "input_mri" for single file inference')
        print('  - Set "input_dir" for batch/directory inference')
        print("\nOr use command line arguments:")
        print("  python inference.py --help")
        print("="*60)
        sys.exit(1)
    
    # Set seed
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    
    # Check device
    device = cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    
    print("\n" + "="*60)
    print("PFGM-FIR CT Volume Inference")
    print("="*60)
    print(f"Checkpoint: {cfg['checkpoint']}")
    print(f"Device: {device}")
    print(f"Sampling steps: {cfg['num_steps']}")
    print(f"CFG scale: {cfg['cfg_scale']}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Num workers: {cfg['num_workers']}")
    print("="*60 + "\n")
    
    # Load model
    model = load_model(cfg["checkpoint"], DEFAULT_MODEL_CONFIG.copy(), device)
    
    # Run inference
    start_time = time.time()
    
    if has_batch:
        # Directory/batch mode (takes priority)
        output_dir = cfg["output_dir"]
        if output_dir is None:
            output_dir = str(Path(cfg["input_dir"]) / "predictions")
            print(f"Auto-generated output directory: {output_dir}")
        
        infer_directory(
            model, cfg["input_dir"], output_dir,
            mr_pattern=cfg["mr_pattern"],
            target_size=cfg["target_size"],
            batch_size=cfg["batch_size"],
            num_steps=cfg["num_steps"],
            sigma_min=DEFAULT_MODEL_CONFIG["sigma_min"],
            sigma_max=DEFAULT_MODEL_CONFIG["sigma_max"],
            sigma_data=DEFAULT_MODEL_CONFIG["sigma_data"],
            rho=DEFAULT_MODEL_CONFIG["rho"],
            cfg_scale=cfg["cfg_scale"],
            device=device,
            num_workers=cfg["num_workers"],
            prefetch_factor=cfg["prefetch_factor"],
        )
    else:
        # Single file mode
        output_ct = cfg["output_ct"]
        if output_ct is None:
            input_path = Path(cfg["input_mri"])
            output_ct = str(input_path.parent / f"{input_path.stem}_pred_ct.nii.gz")
            print(f"Auto-generated output path: {output_ct}")
        
        infer_single(
            model, cfg["input_mri"], output_ct,
            target_size=cfg["target_size"],
            batch_size=cfg["batch_size"],
            num_steps=cfg["num_steps"],
            sigma_min=DEFAULT_MODEL_CONFIG["sigma_min"],
            sigma_max=DEFAULT_MODEL_CONFIG["sigma_max"],
            sigma_data=DEFAULT_MODEL_CONFIG["sigma_data"],
            rho=DEFAULT_MODEL_CONFIG["rho"],
            cfg_scale=cfg["cfg_scale"],
            device=device,
            num_workers=cfg["num_workers"],
            prefetch_factor=cfg["prefetch_factor"],
        )
    
    elapsed = time.time() - start_time
    print(f"\nTotal inference time: {elapsed:.1f}s")


def main():
    """
    Main entry point. Uses config file values if no CLI args provided,
    otherwise uses command line arguments.
    """
    # Check if any meaningful CLI args were provided
    # (sys.argv[0] is the script name, so len > 1 means args were given)
    if len(sys.argv) > 1:
        # Use command line arguments
        args = parse_args()
        
        # Set seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        # Validate input arguments
        if args.input is None and args.input_dir is None:
            print("Error: Must specify either --input or --input_dir")
            sys.exit(1)
        
        if args.input is not None and args.output is None:
            # Auto-generate output name
            input_path = Path(args.input)
            args.output = str(input_path.parent / f"{input_path.stem}_pred_ct.nii.gz")
            print(f"Auto-generated output path: {args.output}")
        
        if args.input_dir is not None and args.output_dir is None:
            args.output_dir = str(Path(args.input_dir) / "predictions")
            print(f"Auto-generated output directory: {args.output_dir}")
        
        # Build model config
        model_config = DEFAULT_MODEL_CONFIG.copy()
        model_config["base_channels"] = args.base_channels
        model_config["channel_mults"] = [int(x) for x in args.channel_mults.split(",")]
        
        # Check device
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            args.device = "cpu"
        
        print("\n" + "="*60)
        print("PFGM-FIR CT Volume Inference")
        print("="*60)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Device: {args.device}")
        print(f"Sampling steps: {args.num_steps}")
        print(f"CFG scale: {args.cfg_scale}")
        print(f"Batch size: {args.batch_size}")
        print(f"Num workers: {args.num_workers}")
        print("="*60 + "\n")
        
        # Load model
        model = load_model(args.checkpoint, model_config, args.device)
        
        # Run inference
        start_time = time.time()
        
        if args.input is not None:
            # Single file mode
            infer_single(
                model, args.input, args.output,
                target_size=args.target_size,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                sigma_data=args.sigma_data,
                rho=args.rho,
                cfg_scale=args.cfg_scale,
                device=args.device,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
            )
        else:
            # Directory mode
            infer_directory(
                model, args.input_dir, args.output_dir,
                mr_pattern=args.mr_pattern,
                target_size=args.target_size,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                sigma_data=args.sigma_data,
                rho=args.rho,
                cfg_scale=args.cfg_scale,
                device=args.device,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
            )
        
        elapsed = time.time() - start_time
        print(f"\nTotal inference time: {elapsed:.1f}s")
    else:
        # No CLI args - use config from file
        print("No command line arguments provided, using INFERENCE_CONFIG from file...")
        run_from_config()


if __name__ == "__main__":
    main()
