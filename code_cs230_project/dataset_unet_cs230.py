import torch
import random
import numpy as np 
from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    ScaleIntensityRanged,
    RandFlipd,
    EnsureTypeD,
    ResizeWithPadOrCropd,
    MapTransform
)

from monai.data import Dataset, MetaTensor

#Custom Transform for Extracting Paired Slices
class ExtractPairedResizeSlicesd(MapTransform):
    
    def __init__(self, keys=("mr", "ct"), 
                 output_keys=("mr_slices", "ct_slices"),
                 num_samples_per_3d_patch=5, 
                 patch_size_2d=(256, 256),
                 allow_missing_keys=False):
        
        super().__init__(keys, allow_missing_keys)

        if len(keys) != 2 or len(output_keys) != 2:
            raise ValueError("keys and output_keys must both contain two elements")
        
        self.mr_key, self.ct_key = keys
        self.mr_slice_key, self.ct_slice_key = output_keys
        self.num_slices = num_samples_per_3d_patch
        self.target_H, self.target_W = patch_size_2d

    def _resize_slice(self, slice_2d_tensor):
        # Helper to resize a single 2D slice
        temp_dict_for_resizer = {"img_slice": slice_2d_tensor}
        resizer = ResizeWithPadOrCropd(keys=["img_slice"], spatial_size=(self.target_H, self.target_W), method="symmetric")
        resized_slice_dict = resizer(temp_dict_for_resizer)
        return resized_slice_dict["img_slice"]
    
    def __call__(self, data: dict) -> dict:
        d = dict(data) # Make a copy to modify
        mr_3d_patch = d.get(self.mr_key)
        ct_3d_patch = d.get(self.ct_key)

        # Default fallback values
        C_mr_fallback, C_ct_fallback = 1, 1
        if hasattr(mr_3d_patch, 'shape') and len(mr_3d_patch.shape) > 0 : C_mr_fallback = mr_3d_patch.shape[0] if mr_3d_patch.ndim > 2 else 1
        if hasattr(ct_3d_patch, 'shape') and len(ct_3d_patch.shape) > 0 : C_ct_fallback = ct_3d_patch.shape[0] if ct_3d_patch.ndim > 2 else 1
        
        dummy_mr_slices = torch.zeros((self.num_slices, C_mr_fallback, self.target_H, self.target_W))
        dummy_ct_slices = torch.zeros((self.num_slices, C_ct_fallback, self.target_H, self.target_W))

        if not all(isinstance(tensor, (torch.Tensor, MetaTensor)) and tensor.ndim == 4 
                   for tensor in [mr_3d_patch, ct_3d_patch]):
            print(f"Warning: MR or CT 3D patch is not a valid 4D tensor for patient ID {d.get('id', 'Unknown')}. "
                  f"MR type: {type(mr_3d_patch)}, CT type: {type(ct_3d_patch)}. Using dummy slices.")
            d[self.mr_slice_key] = dummy_mr_slices
            d[self.ct_slice_key] = dummy_ct_slices
            return d

        C_mr, D_mr, H_mr, W_mr = mr_3d_patch.shape
        C_ct, D_ct, H_ct, W_ct = ct_3d_patch.shape

        if D_mr != D_ct:
            print(f"Warning: MR depth ({D_mr}) and CT depth ({D_ct}) mismatch for patient ID {d.get('id', 'Unknown')}. Using dummy slices.")
            d[self.mr_slice_key] = dummy_mr_slices
            d[self.ct_slice_key] = dummy_ct_slices
            return d
        
        if D_mr == 0:
            print(f"Warning: Zero depth for patches for patient ID {d.get('id', 'Unknown')}. Using dummy slices.")
            d[self.mr_slice_key] = dummy_mr_slices
            d[self.ct_slice_key] = dummy_ct_slices
            return d

        mr_slices_data = []
        ct_slices_data = []
        
        for _ in range(self.num_slices):
            slice_idx = random.randint(0, D_mr - 1) 
            
            mr_slice_2d = mr_3d_patch[:, slice_idx, :, :]
            ct_slice_2d = ct_3d_patch[:, slice_idx, :, :]
            
            resized_mr_slice = self._resize_slice(mr_slice_2d)
            resized_ct_slice = self._resize_slice(ct_slice_2d)
            
            mr_slices_data.append(resized_mr_slice)
            ct_slices_data.append(resized_ct_slice)
        
        if not mr_slices_data or not ct_slices_data: # Safeguard
            d[self.mr_slice_key] = dummy_mr_slices
            d[self.ct_slice_key] = dummy_ct_slices
            return d

        d[self.mr_slice_key] = torch.stack(mr_slices_data) 
        d[self.ct_slice_key] = torch.stack(ct_slices_data) 
        
        return d


# Preprocessing and Dataset for 2D Slices for U-Net
def get_unet_transforms(mode="train", 
                        keys=("mr", "ct"), 
                        patch_size_2d=(256, 256), 
                        num_samples_per_3d_patch=5,
                        mr_input_range_for_scaling=(0, 1.0), 
                        ct_input_range_for_scaling=(0, 1.0)  
                        ):
    

    load_patch = LoadImageD(keys=keys, image_only=True) 
    ensure_channel_first = EnsureChannelFirstD(keys=keys) 

    intensity_scaling = [
        ScaleIntensityRanged(
            keys=["mr"], 
            a_min=mr_input_range_for_scaling[0], 
            a_max=mr_input_range_for_scaling[1],
            b_min=0,                  
            b_max=1.0,                   
            clip=True
        ),
        ScaleIntensityRanged(
            keys=["ct"], 
            a_min=ct_input_range_for_scaling[0], 
            a_max=ct_input_range_for_scaling[1],
            b_min=0, 
            b_max=1.0, 
            clip=True 
        )
    ]
    
    # Use the new custom transform
    slice_extraction_transform = ExtractPairedResizeSlicesd(
        keys=keys, # Original keys it reads from ("mr", "ct")
        output_keys=("mr_slices", "ct_slices"), # New keys it creates
        num_samples_per_3d_patch=num_samples_per_3d_patch,
        patch_size_2d=patch_size_2d
    )
    
    common_transforms = [load_patch, ensure_channel_first] + intensity_scaling
    
    
    transforms_list = common_transforms + [slice_extraction_transform]
    
    transforms_list.insert(0, EnsureTypeD(keys=keys, dtype=torch.float32, data_type="tensor")) # Ensure loaded are tensors
    
    return Compose(transforms_list)

#Slicing each 3D patch into 2D slices for U-Net
class SlicedUNetDataset(Dataset):
    def __init__(self, data, transform, num_samples_per_3d_patch, fallback_patch_size_2d=(256,256)):
        super().__init__(data, transform) 
        self.num_samples_per_3d_patch = num_samples_per_3d_patch
        self.fallback_patch_size_2d = fallback_patch_size_2d

    def __len__(self):
        return len(self.data) * self.num_samples_per_3d_patch

    def _transform(self, index):
        volume_idx = index // self.num_samples_per_3d_patch
        slice_in_volume_idx = index % self.num_samples_per_3d_patch
        
        processed_data_dict = super()._transform(volume_idx)

        item_dict = {}
        
        mr_slices_tensor = processed_data_dict.get("mr_slices")
        ct_slices_tensor = processed_data_dict.get("ct_slices")

        if isinstance(mr_slices_tensor, (torch.Tensor, MetaTensor)) and \
           mr_slices_tensor.ndim == 4 and \
           mr_slices_tensor.shape[0] == self.num_samples_per_3d_patch:
            item_dict["mr"] = mr_slices_tensor[slice_in_volume_idx]
        else:
            print(f"Warning: Could not extract MR slice {slice_in_volume_idx} from volume {volume_idx}. "
                  f"Data for 'mr_slices': {type(mr_slices_tensor)}")
            C = 1 
            H, W = self.fallback_patch_size_2d
            item_dict["mr"] = torch.zeros((C, H, W))

        if isinstance(ct_slices_tensor, (torch.Tensor, MetaTensor)) and \
           ct_slices_tensor.ndim == 4 and \
           ct_slices_tensor.shape[0] == self.num_samples_per_3d_patch:
            item_dict["ct"] = ct_slices_tensor[slice_in_volume_idx]
        else:
            print(f"Warning: Could not extract CT slice {slice_in_volume_idx} from volume {volume_idx}. "
                  f"Data for 'ct_slices': {type(ct_slices_tensor)}")
            C = 1
            H, W = self.fallback_patch_size_2d
            item_dict["ct"] = torch.zeros((C, H, W))

        if "id" in processed_data_dict: 
            item_dict["id"] = f"{processed_data_dict['id']}_slice{slice_in_volume_idx}"
        elif isinstance(self.data[volume_idx], dict) and "id" in self.data[volume_idx]:
             item_dict["id"] = f"{self.data[volume_idx]['id']}_slice{slice_in_volume_idx}"
        else: 
            item_dict["id"] = f"vol{volume_idx}_slice{slice_in_volume_idx}"
            
        
            
        return item_dict