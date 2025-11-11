import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import monai
from monai.utils import first, set_determinism
import glob
import numpy as np
import nibabel as nib 
import matplotlib.pyplot as plt 
import random



from model_unet_cs230 import UNet 
from dataset_unet_cs230 import get_unet_transforms as get_transforms
from dataset_unet_cs230 import SlicedUNetDataset as SlicedDataset
# --- Helper Functions ---
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Training Script ---
def main_unet_train(config):
    device = get_device()
    set_determinism(seed=config.get("seed", 0)) 
    print(f"Using device: {device}")
    
    # Data loading (expects paired MR and CT preprocessed 3D patches)
    
    
    all_files = []
    # Iterate through patient folders in the processed data root
    for patient_dir in sorted(glob.glob(os.path.join(config["PROCESSED_DATA_ROOT"], "*"))):
        mr_patch_path = glob.glob(os.path.join(patient_dir, f"{config['MR_INPUT_SUFFIX']}"))
        ct_patch_path = glob.glob(os.path.join(patient_dir, f"{config['CT_INPUT_SUFFIX']}"))

        if mr_patch_path and ct_patch_path: # Ensure both MR and CT patches exist for the patient
            all_files.append({
                "mr": mr_patch_path[0], 
                "ct": ct_patch_path[0],
                "id": os.path.basename(patient_dir)
            })
        else:
            print(f"Warning: Missing MR or CT patch for {os.path.join(patient_dir, f"{config['MR_INPUT_SUFFIX']}")}")
            print(f"Warning: Missing MR or CT patch for {os.path.basename(patient_dir)}. Skipping.")

    if not all_files:
        print("Error: No paired MR/CT preprocessed patches found. Please check PROCESSED_DATA_ROOT and suffixes.")
        return
        
    print(f"Found {len(all_files)} paired MR/CT 3D patches.")

    # Split data (example: 80% train, 20% val - adjust as needed)
    num_total = len(all_files)
    num_train = int(num_total * 0.8)
    
    
    random.Random(config.get("seed", 0)).shuffle(all_files) 
    
    train_files_dicts = all_files[:num_train]
    val_files_dicts = all_files[num_train:]

    print(f"Training with {len(train_files_dicts)} 3D patches, Validating with {len(val_files_dicts)} 3D patches.")

    if not train_files_dicts:
        print("Error: No training files after split. Increase dataset size or adjust split.")
        return

    train_transforms = get_transforms(
        mode="train", 
        keys=("mr", "ct"), 
        patch_size_2d=config["patch_size_2d"],
        num_samples_per_3d_patch=config["num_slices_per_3d_patch"]
    )
    
    val_transforms = get_transforms(
        mode="val", 
        keys=("mr", "ct"), 
        patch_size_2d=config["patch_size_2d"],
        num_samples_per_3d_patch=config.get("num_val_slices_per_3d_patch", 1)
    )

    train_ds = SlicedDataset(data=train_files_dicts, transform=train_transforms, 
                                 num_samples_per_3d_patch=config["num_slices_per_3d_patch"],
                                 fallback_patch_size_2d=config["patch_size_2d"])
    
    if len(val_files_dicts) > 0:
        val_ds = SlicedDataset(data=val_files_dicts, transform=val_transforms,
                                   num_samples_per_3d_patch=config.get("num_val_slices_per_3d_patch", 1),
                                   fallback_patch_size_2d=config["patch_size_2d"])
        val_loader = DataLoader(val_ds, batch_size=config["batch_size_val"], shuffle=False, num_workers=config["num_workers_loader"])
    else:
        val_loader = None
        print("No validation data. Skipping validation loop.")

    if len(train_ds) == 0:
        print(f"Error: Training dataset is empty after SlicedDataset initialization.")
        return
        
    train_loader = DataLoader(train_ds, batch_size=config["batch_size_train"], shuffle=True, num_workers=config["num_workers_loader"], drop_last=True)

    # --- Model Initialization ---
    model = UNet(
        n_channels_in=1, 
        n_channels_out=1, 
        base_filters=config.get("unet_base_filters", 64),
        bilinear=config.get("unet_bilinear_upsampling", True)
    ).to(device)
    
    
    
    if config.get("unet_final_activation_tanh", False): # Check if this key exists in config
        model.outc = nn.Sequential(model.outc, nn.Tanh()) # Append Tanh to the output convolution
        print("Added Tanh activation to U-Net output.")

    if config.get("unet_final_activation_sigmoid", False): # Check if this key exists in config
        model.outc = nn.Sequential(model.outc, nn.Sigmoid()) # Append Sigmoid to the output convolution
        print("Added Sigmoid activation to U-Net output.")

    # --- Loss Function and Optimizer ---
    if config.get("loss_function", "L1").upper() == "L1":
        criterion = nn.L1Loss().to(device) # Mean Absolute Error
        print("Using L1 Loss (MAE).")
    elif config.get("loss_function", "MSE").upper() == "MSE":
        criterion = nn.MSELoss().to(device) # Mean Squared Error
        print("Using MSE Loss.")
    elif config.get("loss_function", "L1+MSE").upper() == "L1+MSE":
        criterion1 =  nn.MSELoss().to(device)  # Mean Squared Error
        criterion2 = nn.L1Loss().to(device) # Mean Absolute Error
        criterion = lambda output, target: config.get('comb_loss', 0.75) * criterion1(output, target) + (1 - config.get('comb_loss', 0.75)) * criterion2(output, target)
        print("Using Combined Loss (L1 + MSE) with weight:", config.get('comb_loss', 0.5))
    else:
        raise ValueError(f"Unsupported loss_function: {config.get('loss_function')}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    if config.get("use_lr_scheduler", False):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.get("lr_scheduler_step_size", 30), gamma=config.get("lr_scheduler_gamma", 0.1))
        print(f"Using StepLR scheduler: step_size={config.get('lr_scheduler_step_size',30)}, gamma={config.get('lr_scheduler_gamma',0.1)}")
    else:
        scheduler = None

    # --- Training Loop ---
    history_train_loss = []
    history_val_loss = []

    print("Starting U-Net training...")
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_train_loss_accum = 0.0
        
        for i, batch_data in enumerate(train_loader):
            if "mr" not in batch_data or "ct" not in batch_data:
                print(f"Skipping training batch {i} due to missing 'mr' or 'ct' keys.")
                continue
            
            inputs_mr = batch_data["mr"].to(device) # Input MR slice
            targets_ct = batch_data["ct"].to(device) # Target CT slice

            optimizer.zero_grad()
            outputs_sct = model(inputs_mr) # Synthesized CT slice
            
            loss = criterion(outputs_sct, targets_ct)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss_accum += loss.item()

            if (i + 1) % config["log_interval"] == 0:
                print(
                    f"Epoch [{epoch+1}/{config['num_epochs']}], Batch [{i+1}/{len(train_loader)}] | "
                    f"Train Loss: {loss.item():.4f}"
                )
        
        avg_epoch_train_loss = epoch_train_loss_accum / len(train_loader) if len(train_loader) > 0 else 0
        history_train_loss.append(avg_epoch_train_loss)
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Avg Training Loss: {avg_epoch_train_loss:.4f}")

        # Validation loop
        if val_loader and (epoch + 1) % config.get("val_interval", 1) == 0:
            model.eval()
            epoch_val_loss_accum = 0.0
            with torch.no_grad():
                for val_batch_data in val_loader:
                    if "mr" not in val_batch_data or "ct" not in val_batch_data:
                        print(f"Skipping validation batch due to missing 'mr' or 'ct' keys.")
                        continue
                    val_inputs_mr = val_batch_data["mr"].to(device)
                    val_targets_ct = val_batch_data["ct"].to(device)
                    
                    val_outputs_sct = model(val_inputs_mr)
                    val_loss = criterion(val_outputs_sct, val_targets_ct)
                    epoch_val_loss_accum += val_loss.item()
            
            avg_epoch_val_loss = epoch_val_loss_accum / len(val_loader) if len(val_loader) > 0 else 0
            history_val_loss.append(avg_epoch_val_loss)
            print(f"Avg Validation Loss: {avg_epoch_val_loss:.4f}")
        elif val_loader: # If val_loader exists but not validation interval, append last val loss or NaN
             history_val_loss.append(history_val_loss[-1] if history_val_loss else float('nan'))


        if scheduler:
            scheduler.step()
            print(f"LR Scheduler stepped. Current LR: {scheduler.get_last_lr()[0]:.6f}")


        if (epoch + 1) % config["save_interval"] == 0:
            os.makedirs(config["output_dir"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config["output_dir"], f"unet_model_epoch_{epoch+1}.pth"))
            print(f"Saved U-Net model at epoch {epoch+1}")
            
    print("U-Net training finished.")

    # --- Plotting and Saving Losses ---
    if config["num_epochs"] > 0 :
        plt.figure(figsize=(10, 5))
        plt.plot(history_train_loss, label="Training Loss")
        if history_val_loss and not all(np.isnan(history_val_loss)): # Plot val loss if available
            plt.plot(history_val_loss, label="Validation Loss")
        plt.title("U-Net Training Loss Evolution")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (e.g., L1 or MSE)")
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(config["output_dir"], "unet_loss_evolution.png")
        plt.savefig(loss_plot_path)
        print(f"Saved U-Net loss evolution plot to {loss_plot_path}")


if __name__ == "__main__":
    
    # global config_unet 
    config_unet = {
        "PROCESSED_DATA_ROOT": "", 
        "MR_INPUT_SUFFIX": "mr.nii_processed_patch.nii.gz", 
        "CT_INPUT_SUFFIX": "ct.nii_processed_patch.nii.gz", 
        "output_dir": "", 
        "patch_size_2d": (256, 256), 
        "num_slices_per_3d_patch": 10, 
        "num_val_slices_per_3d_patch": 5, # Fewer slices for faster validation
        "mr_input_range": (0, 1.0), # Assuming MR patches are already effectively in this range
        "ct_input_range": (0, 1.0), # Assuming CT patches are already in this range
        "batch_size_train": 15, 
        "batch_size_val": 2,
        "comb_loss": 0.75,
        "num_workers_loader": 0, 
        "learning_rate": 1e-4, 
        "num_epochs": 15, 
        "unet_base_filters": 64,
        "unet_bilinear_upsampling": True,
        "unet_final_activation_tanh": False,
        "unet_final_activation_sigmoid": True, 
        "loss_function": "MSE", 
        "use_lr_scheduler": True,
        "lr_scheduler_step_size": 20,
        "lr_scheduler_gamma": 0.5,
        "log_interval": 1,    
        "save_interval": 1,   
        "val_interval": 1, # Validate every epoch
        "seed": 42
    }
    

    main_unet_train(config_unet)
