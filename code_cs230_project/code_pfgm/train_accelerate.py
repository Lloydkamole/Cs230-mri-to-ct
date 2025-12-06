"""
PFGM-FIR Training Script with Hugging Face Accelerate.

Multi-GPU training with FSDP support for larger models.

Usage:
    # Single GPU:
    python train_accelerate.py
    
    # Multi-GPU with accelerate:
    accelerate launch --num_processes 2 --gpu_ids 0,1 train_accelerate.py
    
    # Or with config file:
    accelerate launch --config_file accelerate_config.yaml train_accelerate.py
"""

import os
import sys
import math
import time
import random
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torchvision.utils as vutils

# Hugging Face Accelerate
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed, ProjectConfiguration

# Metrics
try:
    import torchmetrics
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not installed. Metrics will be disabled.")

# Import loader
from loader import MonaiNiftiSliceDataset, get_monai_nifti_transforms

# Import model
from models import create_swin_fir_unet

# Compatibility fix for Pillow >= 10
try:
    from PIL import Image as _PILImage
    if not hasattr(_PILImage, "ANTIALIAS"):
        _PILImage.ANTIALIAS = _PILImage.Resampling.LANCZOS
except Exception:
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================

config = {
    # Data Settings
    "data_dir": "/home/lloyd/Documents/AR PFGM/train_pfgm",
    "mr_pattern": "mr.nii", # Suffix pattern to match MRI files
    "ct_pattern": "ct.nii", # Suffix pattern to match CT files
    "train_split": 0.9,
    "image_size": 256,
    
    # Model Architecture
    "model_config": "custom", # "custom" or predefined config name
    "base_channels": 64,   
    "channel_mults": [1, 2, 4, 8],
    "num_res_blocks": 3,
    "attention_resolutions": [2, 4, 8],  # Downsample factors for attention
    "num_heads": 4,
    "mlp_ratio": 4.0, # MLP expansion ratio in transformer blocks
    "radius_emb_dim": 256,
    "window_size": 4, # Swin Transformer window size
    "use_sfb": True,  # Use Shifted-Fixed Blocks
    "dropout": 0.0,
    "drop_path": 0.1,
    
    # EDM Parameters
    "sigma_min": 0.002,
    "sigma_max": 80.0,
    "sigma_data": 0.5,
    "P_mean": -1.2,
    "P_std": 1.2,
    "rho": 7.0,
    "loss_type": "l2",
    
    # Classifier-Free Guidance (CFG)
    "cond_drop_prob": 0.2,               # Probability of dropping MRI conditioning (0.0 = no CFG)
    "cfg_scale": 2.0,                    # Guidance scale at inference (1.0 = no guidance)
    
    # Training Parameters
    "num_epochs": 100,
    "batch_size": 3,  # Per GPU batch size (reduced for multi-GPU memory)
    "lr": 1e-4,
    "weight_decay": 0.01,
    "grad_clip": None,
    "warmup_steps": 500,
    
    # Sampling / Generation
    "num_sample_steps": 50,
    "num_val_samples": 4,
    
    # Logging & Checkpointing
    "output_dir": "./outputs_accelerate",
    "log_every": 5,
    "val_every": 1,
    "save_every": 5,
    "sample_every": 1,
    "resume": None,
    
    # Gradient & ODE Monitoring
    "log_gradients": True,
    "log_ode_trajectory": True,
    "grad_log_interval": 100,
    
    # System
    "seed": 42,
    "num_workers": 4,
    "mixed_precision": "no",  # "no", "fp16", or "bf16"
}


# =============================================================================
# Utility Functions
# =============================================================================

def collect_data_files(data_path: str, mr_pattern: str = "mr.nii", ct_pattern: str = "ct.nii"):
    """Collect MR and CT file pairs from directory structure."""
    data_files = []
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    for patient_dir in sorted(os.listdir(data_path)):
        patient_path = os.path.join(data_path, patient_dir)
        if not os.path.isdir(patient_path):
            continue
        
        mr_file = None
        ct_file = None
        for filename in os.listdir(patient_path):
            if mr_pattern in filename:
                mr_file = os.path.join(patient_path, filename)
            if ct_pattern in filename:
                ct_file = os.path.join(patient_path, filename)
        
        if mr_file and ct_file:
            data_files.append({"mr": mr_file, "ct": ct_file, "id": patient_dir})
    
    return data_files


# =============================================================================
# Gradient & ODE Monitoring (from train_pfgm_fir.py)
# =============================================================================

def check_gradient_flow(named_parameters):
    """
    Check gradient flow through network layers.
    
    Returns dict with gradient statistics per layer for monitoring.
    Useful for detecting vanishing/exploding gradients.
    """
    grad_info = {}
    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            grad_info[name] = {
                'mean': param.grad.abs().mean().item(),
                'max': param.grad.abs().max().item(),
                'has_nan': torch.isnan(param.grad).any().item(),
            }
    return grad_info


# =============================================================================
# EDM Functions
# =============================================================================

def edm_precond(
    model: nn.Module,
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


def sample_sigma(
    batch_size: int,
    P_mean: float = -1.2,
    P_std: float = 1.2,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Sample sigma values according to EDM log-normal distribution."""
    rnd_normal = torch.randn(batch_size, device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    sigma = sigma.clamp(min=sigma_min, max=sigma_max)
    return sigma


def edm_loss(
    model: nn.Module,
    x_clean: torch.Tensor,
    mri: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
    loss_type: str = 'l2',
    cond_drop_prob: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute EDM-style loss with optional classifier-free guidance training.
    
    Args:
        cond_drop_prob: Probability of dropping MRI conditioning (replacing with zeros).
                        Set to 0.0 to disable CFG training.
    
    Returns:
        loss: Scalar loss value
        metrics: Dictionary of metrics for logging
    """
    n = torch.randn_like(x_clean) * sigma.view(-1, 1, 1, 1)
    x_noisy = x_clean + n
    
    # Classifier-free guidance: randomly drop conditioning
    if cond_drop_prob > 0.0 and model.training:
        # Create mask for which samples drop conditioning
        drop_mask = torch.rand(mri.shape[0], device=mri.device) < cond_drop_prob
        # Zero out MRI for dropped samples
        mri = mri.clone()
        mri[drop_mask] = 0.0
    
    D_x = edm_precond(model, x_noisy, mri, sigma, sigma_data)
    
    sigma_v = sigma.view(-1, 1, 1, 1)
    weight = (sigma_v ** 2 + sigma_data ** 2) / ((sigma_v * sigma_data) ** 2)
    weight = weight.clamp(max=100.0)
    
    if loss_type == 'l1':
        loss = (weight * (D_x - x_clean).abs()).mean()
    elif loss_type == 'l2':
        loss = (weight * (D_x - x_clean) ** 2).mean()
    elif loss_type == 'huber':
        loss = (weight * F.smooth_l1_loss(D_x, x_clean, reduction='none')).mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Metrics for logging (matching train_pfgm_fir.py)
    with torch.no_grad():
        mse = F.mse_loss(D_x, x_clean).item()
        pred_std = D_x.std().item()
        target_std = x_clean.std().item()
    
    metrics = {
        'mse': mse,
        'pred_std': pred_std,
        'target_std': target_std,
        'sigma_mean': sigma.mean().item(),
        'sigma_max': sigma.max().item(),
        'sigma_min': sigma.min().item(),
    }
    
    return loss, metrics


@torch.no_grad()
def edm_sample(
    model: nn.Module,
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
    return_trajectory_stats: bool = False,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """
    Generate CT images from MRI using EDM sampling (Algorithm 2 from EDM paper).
    
    This is a deterministic ODE sampler with optional stochasticity.
    
    Args:
        return_trajectory_stats: If True, returns (samples, stats_dict) with ODE trajectory info
        cfg_scale: Classifier-free guidance scale. 1.0 = no guidance, >1.0 = stronger conditioning.
                   Requires model trained with cond_drop_prob > 0.
    """
    B, C, H, W = mri.shape
    device = mri.device
    
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    
    x_next = torch.randn(B, 1, H, W, device=device, dtype=torch.float64) * t_steps[0]
    
    # Track ODE trajectory statistics
    traj_norms = [x_next.norm().item()]
    denoised_norms = []
    score_norms = []
    update_norms = []
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # Increase noise temporarily (stochasticity)
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        
        sigma_batch = torch.full((B,), t_hat.item(), device=device, dtype=torch.float32)
        denoised_cond = edm_precond(model, x_hat.float(), mri, sigma_batch, sigma_data).double()
        
        # Classifier-free guidance
        if cfg_scale != 1.0:
            mri_uncond = torch.zeros_like(mri)
            denoised_uncond = edm_precond(model, x_hat.float(), mri_uncond, sigma_batch, sigma_data).double()
            denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
        else:
            denoised = denoised_cond
        
        d_cur = (x_hat - denoised) / t_hat
        update = (t_next - t_hat) * d_cur
        x_next = x_hat + update
        
        if i < num_steps - 1 and t_next > 0:
            sigma_batch = torch.full((B,), t_next.item(), device=device, dtype=torch.float32)
            denoised_cond_next = edm_precond(model, x_next.float(), mri, sigma_batch, sigma_data).double()
            
            # CFG for Heun correction
            if cfg_scale != 1.0:
                denoised_uncond_next = edm_precond(model, x_next.float(), mri_uncond, sigma_batch, sigma_data).double()
                denoised_next = denoised_uncond_next + cfg_scale * (denoised_cond_next - denoised_uncond_next)
            else:
                denoised_next = denoised_cond_next
            
            d_prime = (x_next - denoised_next) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        # Track statistics at checkpoints (every ~20% of steps)
        checkpoint_interval = max(1, num_steps // 5)
        if i % checkpoint_interval == 0 or i == num_steps - 2:
            traj_norms.append(x_next.norm().item())
            denoised_norms.append(denoised.norm().item())
            score_norms.append(d_cur.norm().item())
            update_norms.append(update.norm().item())
    
    result = x_next.float()
    
    if return_trajectory_stats:
        stats = {
            'traj_norms': traj_norms,
            'denoised_norms': denoised_norms,
            'score_norms': score_norms,
            'update_norms': update_norms,
            'initial_norm': traj_norms[0],
            'final_norm': traj_norms[-1],
            'mean_denoised_norm': np.mean(denoised_norms) if denoised_norms else 0,
            'mean_score_norm': np.mean(score_norms) if score_norms else 0,
            'mean_update_norm': np.mean(update_norms) if update_norms else 0,
        }
        return result, stats
    
    return result


# =============================================================================
# Training Function
# =============================================================================

def train(cfg: dict):
    """Main training function with Accelerate."""
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg["output_dir"]) / f"{timestamp}_pfgm_fir"
    
    # Initialize Accelerator
    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
    )
    
    accelerator = Accelerator(
        mixed_precision=cfg["mixed_precision"],
        gradient_accumulation_steps=1,
        project_config=project_config,
        log_with="tensorboard",
    )
    
    # Set seed
    set_seed(cfg["seed"])
    
    # Only main process prints
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"PFGM-FIR Training with Accelerate")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Mixed precision: {cfg['mixed_precision']}")
        print(f"{'='*60}\n")
    
    # Build model
    if cfg["model_config"] == "custom":
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
    else:
        model = create_swin_fir_unet(
            config=cfg["model_config"],
            in_channels=2,
            out_channels=1,
        )
    
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params/1e6:.2f}M")
        print(f"  Architecture config:")
        print(f"    base_channels: {cfg['base_channels']}")
        print(f"    channel_mults: {cfg['channel_mults']}")
        print(f"    num_res_blocks: {cfg['num_res_blocks']}")
        print(f"    attention_resolutions: {cfg['attention_resolutions']}")
        print(f"    num_heads: {cfg['num_heads']}")
    
    # Check if FSDP is enabled (detected from accelerate config)
    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    if accelerator.is_main_process:
        print(f"Distributed type: {accelerator.distributed_type}")
        if is_fsdp:
            print("FSDP enabled - optimizer will be created after model wrapping")
    
    # For non-FSDP, create optimizer before prepare (standard path)
    optimizer = None
    if not is_fsdp:
        optimizer = AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(0.9, 0.95),
        )
    
    # Data - only main process loads initially to avoid duplicate scanning
    if accelerator.is_main_process:
        print("Loading data files...")
    
    all_files = collect_data_files(
        cfg["data_dir"],
        mr_pattern=cfg["mr_pattern"],
        ct_pattern=cfg["ct_pattern"]
    )
    
    random.shuffle(all_files)
    split_idx = max(1, int(len(all_files) * cfg["train_split"]))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:] if len(all_files) > split_idx else None
    
    if accelerator.is_main_process:
        print(f"Found {len(train_files)} train, {len(val_files) if val_files else 0} val patients")
        print("Creating datasets...")
    
    train_transforms = get_monai_nifti_transforms(mode="train", target_size=cfg["image_size"])
    train_dataset = MonaiNiftiSliceDataset(
        data=train_files,
        transform=train_transforms,
        target_size=cfg["image_size"]
    )
    
    if accelerator.is_main_process:
        print(f"Train dataset: {len(train_dataset)} slices")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    
    if accelerator.is_main_process:
        print(f"Train loader: {len(train_loader)} batches")
    
    val_loader = None
    if val_files:
        val_transforms = get_monai_nifti_transforms(mode="val", target_size=cfg["image_size"])
        val_dataset = MonaiNiftiSliceDataset(
            data=val_files,
            transform=val_transforms,
            target_size=cfg["image_size"]
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )
    
    # Learning rate scheduler (lr_lambda defined here, scheduler created after prepare for FSDP)
    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_steps = cfg.get("warmup_steps", 500)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress))) # Decay to 1% of LR
    
    # For non-FSDP, create scheduler before prepare
    scheduler = None
    if not is_fsdp:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Prepare with Accelerate
    if accelerator.is_main_process:
        print("Preparing model and data with Accelerator...")
    
    if is_fsdp:
        # FSDP path: prepare model first, then create optimizer from wrapped model
        model = accelerator.prepare(model)
        
        # Create optimizer after FSDP wrapping (critical for FSDP)
        optimizer = AdamW(
            model.parameters(),  # These are now FSDP-wrapped parameters
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(0.9, 0.95),
        )
        
        # Create scheduler with the new optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Prepare optimizer, dataloader, scheduler
        optimizer, train_loader, scheduler = accelerator.prepare(
            optimizer, train_loader, scheduler
        )
    else:
        # Standard path: prepare everything together
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
    
    if accelerator.is_main_process:
        print("Model and data prepared!")
    
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    
    # Initialize tracking (convert non-primitive types to strings for TensorBoard)
    trackable_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, (list, tuple)):
            trackable_cfg[k] = str(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            trackable_cfg[k] = v if v is not None else "None"
        else:
            trackable_cfg[k] = str(v)
    accelerator.init_trackers("pfgm_fir", config=trackable_cfg)
    
    if accelerator.is_main_process:
        print(f"Training batches per epoch: {len(train_loader)}")
        print(f"Total training steps: {total_steps}")
        print(f"Starting training...\n")
    
    # Metrics
    if TORCHMETRICS_AVAILABLE and accelerator.is_main_process:
        psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(accelerator.device)
        ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(accelerator.device)
        mae_metric = torchmetrics.MeanAbsoluteError().to(accelerator.device)
    else:
        psnr_metric = ssim_metric = mae_metric = None
    
    global_step = 0
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(cfg["num_epochs"]):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg['num_epochs']}",
            disable=not accelerator.is_main_process
        )
        
        for batch_idx, batch in enumerate(pbar):
            mr = batch["mr"]
            ct = batch["ct"]
            
            batch_size = mr.shape[0]
            sigma = sample_sigma(
                batch_size=batch_size,
                P_mean=cfg["P_mean"],
                P_std=cfg["P_std"],
                sigma_min=cfg["sigma_min"],
                sigma_max=cfg["sigma_max"],
                device=accelerator.device,
            )
            
            with accelerator.accumulate(model):
                loss, metrics = edm_loss(
                    model, ct, mr, sigma,
                    sigma_data=cfg["sigma_data"],
                    loss_type=cfg["loss_type"],
                    cond_drop_prob=cfg.get("cond_drop_prob", 0.0),
                )
                
                accelerator.backward(loss)
                
                if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                    accelerator.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.3e}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            if batch_idx % cfg["log_every"] == 0:
                accelerator.log({
                    "train/loss": loss.item(),
                    "train/mse": metrics['mse'],
                    "train/pred_std": metrics['pred_std'],
                    "train/target_std": metrics['target_std'],
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/sigma_mean": metrics['sigma_mean'],
                    "train/sigma_max": metrics['sigma_max'],
                    "train/sigma_min": metrics['sigma_min'],
                }, step=global_step)
            
            # Check gradient flow periodically
            if cfg.get("log_gradients", False) and batch_idx % cfg.get("grad_log_interval", 100) == 0:
                unwrapped = accelerator.unwrap_model(model)
                grad_info = check_gradient_flow(unwrapped.named_parameters())
                if grad_info:
                    grad_means = [v['mean'] for v in grad_info.values()]
                    grad_maxs = [v['max'] for v in grad_info.values()]
                    has_nan = any(v['has_nan'] for v in grad_info.values())
                    accelerator.log({
                        "grad/mean_avg": float(np.mean(grad_means)),
                        "grad/mean_max": float(np.max(grad_means)),
                        "grad/max_avg": float(np.mean(grad_maxs)),
                        "grad/max_max": float(np.max(grad_maxs)),
                        "grad/has_nan": float(has_nan),
                    }, step=global_step)
            
            global_step += 1
        
        # Epoch stats
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start if 'epoch_start' in dir() else 0
        
        accelerator.log({
            "train/epoch_loss": avg_loss,
            "system/epoch_time": epoch_time,
        }, step=epoch)
        
        # Log GPU memory if available
        if torch.cuda.is_available() and accelerator.is_main_process:
            gpu_mem_mb = torch.cuda.max_memory_allocated(accelerator.device) / 1024**2
            accelerator.log({"system/gpu_memory_MB": gpu_mem_mb}, step=epoch)
        
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}: avg_loss={avg_loss:.3e}, time={epoch_time:.1f}s")
        
        # Sample generation
        if (epoch + 1) % cfg["sample_every"] == 0:
            model.eval()
            
            # All ranks must participate in DataLoader iteration to avoid NCCL timeout
            sample_batch = next(iter(train_loader))
            mr_samples = sample_batch["mr"][:cfg["num_val_samples"]]
            ct_samples = sample_batch["ct"][:cfg["num_val_samples"]]
            
            # Only main process does generation and logging
            if accelerator.is_main_process:
                with torch.no_grad():
                
                    # Get unwrapped model for sampling
                    unwrapped_model = accelerator.unwrap_model(model)
                    # Sample with optional ODE trajectory logging
                    if cfg.get("log_ode_trajectory", False):
                        pred_ct, ode_stats = edm_sample(
                            unwrapped_model, mr_samples,
                            num_steps=cfg["num_sample_steps"],
                            sigma_min=cfg["sigma_min"],
                            sigma_max=cfg["sigma_max"],
                            sigma_data=cfg["sigma_data"],
                            rho=cfg["rho"],
                            return_trajectory_stats=True,
                            cfg_scale=cfg.get("cfg_scale", 1.0),
                        )
                        # Log ODE trajectory stats
                        accelerator.log({
                            "ode/initial_norm": ode_stats['initial_norm'],
                            "ode/final_norm": ode_stats['final_norm'],
                            "ode/mean_denoised_norm": ode_stats['mean_denoised_norm'],
                            "ode/mean_score_norm": ode_stats['mean_score_norm'],
                            "ode/mean_update_norm": ode_stats['mean_update_norm'],
                        }, step=epoch)
                    else:
                        pred_ct = edm_sample(
                            unwrapped_model, mr_samples,
                            num_steps=cfg["num_sample_steps"],
                            sigma_min=cfg["sigma_min"],
                            sigma_max=cfg["sigma_max"],
                            sigma_data=cfg["sigma_data"],
                            rho=cfg["rho"],
                            cfg_scale=cfg.get("cfg_scale", 1.0),
                        )
                    
                    pred_ct = torch.clamp(pred_ct, -1.0, 1.0)
                    pred_ct_01 = (pred_ct + 1.0) / 2.0
                    ct_samples_01 = (ct_samples + 1.0) / 2.0
                    
                    if psnr_metric is not None:
                        psnr_val = psnr_metric(pred_ct_01, ct_samples_01)
                        ssim_val = ssim_metric(pred_ct_01, ct_samples_01)
                        mae_val = mae_metric(pred_ct_01, ct_samples_01) if mae_metric else None
                        
                        log_dict = {
                            "gen/psnr": psnr_val.item(),
                            "gen/ssim": ssim_val.item(),
                        }
                        if mae_val is not None:
                            log_dict["gen/mae"] = mae_val.item()
                        
                        accelerator.log(log_dict, step=epoch)
                        mae_str = f", MAE={mae_val:.5f}" if mae_val else ""
                        print(f"  Generation: PSNR={psnr_val:.3f}, SSIM={ssim_val:.5f}{mae_str}")
                    
                    # Log training sample images to TensorBoard
                    img_grid = vutils.make_grid(
                        torch.cat([mr_samples.cpu(), pred_ct.cpu(), ct_samples.cpu()], dim=0),
                        nrow=cfg["num_val_samples"],
                        normalize=True,
                        scale_each=True
                    )
                    # Get TensorBoard writer from Accelerate tracker
                    tb_tracker = accelerator.get_tracker("tensorboard")
                    if tb_tracker is not None:
                        tb_tracker.writer.add_image("Gen/MRI_Pred_GT", img_grid, epoch)
            model.train()
        
        # Validation
        if val_loader is not None and (epoch + 1) % cfg["val_every"] == 0:
            model.eval()
            val_loss_accum = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    mr = batch["mr"]
                    ct = batch["ct"]
                    
                    sigma = sample_sigma(
                        batch_size=ct.shape[0],
                        P_mean=cfg["P_mean"],
                        P_std=cfg["P_std"],
                        sigma_min=cfg["sigma_min"],
                        sigma_max=cfg["sigma_max"],
                        device=accelerator.device,
                    )
                    
                    loss, _ = edm_loss(
                        model, ct, mr, sigma,
                        sigma_data=cfg["sigma_data"],
                        loss_type=cfg["loss_type"]
                    )
                    val_loss_accum += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss_accum / val_batches
            accelerator.log({"val/loss": avg_val_loss}, step=epoch)
            
            if accelerator.is_main_process:
                print(f"  Validation: loss={avg_val_loss:.3e}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                accelerator.save_state(output_dir / "checkpoints" / "best")
                if accelerator.is_main_process:
                    print(f"  New best model saved!")
            
            # Validation sample generation
            # All ranks must participate in DataLoader iteration to avoid NCCL timeout
            val_sample_batch = next(iter(val_loader))
            mr_val_samples = val_sample_batch["mr"][:cfg["num_val_samples"]]
            ct_val_samples = val_sample_batch["ct"][:cfg["num_val_samples"]]
            
            # Only main process does generation and logging
            if accelerator.is_main_process and psnr_metric is not None:
                unwrapped_model = accelerator.unwrap_model(model)
                
                # Generate samples from validation MRI
                pred_val_ct = edm_sample(
                    unwrapped_model, mr_val_samples,
                    num_steps=cfg["num_sample_steps"],
                    sigma_min=cfg["sigma_min"],
                    sigma_max=cfg["sigma_max"],
                    sigma_data=cfg["sigma_data"],
                    rho=cfg["rho"],
                    cfg_scale=cfg.get("cfg_scale", 1.0),
                )
                
                pred_val_ct = torch.clamp(pred_val_ct, -1.0, 1.0)
                pred_val_ct_01 = (pred_val_ct + 1.0) / 2.0
                ct_val_samples_01 = (ct_val_samples + 1.0) / 2.0
                
                # Compute validation generation metrics
                val_psnr = psnr_metric(pred_val_ct_01, ct_val_samples_01)
                val_ssim = ssim_metric(pred_val_ct_01, ct_val_samples_01)
                val_mae = mae_metric(pred_val_ct_01, ct_val_samples_01) if mae_metric else None
                
                val_gen_log = {
                    "val_gen/psnr": val_psnr.item(),
                    "val_gen/ssim": val_ssim.item(),
                }
                if val_mae is not None:
                    val_gen_log["val_gen/mae"] = val_mae.item()
                
                accelerator.log(val_gen_log, step=epoch)
                
                val_mae_str = f", MAE={val_mae:.5f}" if val_mae else ""
                print(f"  Val Generation: PSNR={val_psnr:.3f}, SSIM={val_ssim:.5f}{val_mae_str}")
                
                # Log validation sample images to TensorBoard
                val_img_grid = vutils.make_grid(
                    torch.cat([mr_val_samples.cpu(), pred_val_ct.cpu(), ct_val_samples.cpu()], dim=0),
                    nrow=cfg["num_val_samples"],
                    normalize=True,
                    scale_each=True
                )
                tb_tracker = accelerator.get_tracker("tensorboard")
                if tb_tracker is not None:
                    tb_tracker.writer.add_image("Val_Gen/MRI_Pred_GT", val_img_grid, epoch)
            
            model.train()
        
        # Save checkpoint
        if (epoch + 1) % cfg["save_every"] == 0:
            accelerator.save_state(output_dir / "checkpoints" / f"epoch_{epoch+1:04d}")
            if accelerator.is_main_process:
                print(f"  Checkpoint saved")
    
    # Final save
    accelerator.save_state(output_dir / "checkpoints" / "final")
    accelerator.end_training()
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Output: {output_dir}")
        print(f"Best val loss: {best_val_loss:.3e}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    train(config)
