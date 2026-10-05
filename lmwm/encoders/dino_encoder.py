import os

import torch
import torch.nn as nn
from typing import Optional

class DinoV2EncoderPt(nn.Module):
    """Frozen DINOv2 vision encoder (loaded from torch.hub)."""
    
    def __init__(
        self, 
        model_name='dinov2_vits14',
        freeze=True,
        img_norm_type="imagenet"
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.img_norm_type = img_norm_type
        
        # Load DINOv2 from torch.hub
        self.dino_model = torch.hub.load('facebookresearch/dinov2', model_name)
        
        # Get feature dimension from the model
        # dinov2_vits14 has 384 dimensions
        self.num_features = self.dino_model.embed_dim
        
        # Freeze DINO weights
        if self.freeze:
            for param in self.dino_model.parameters():
                param.requires_grad = False
            self.dino_model.eval()
    
    def normalize_images(self, img):
        """Normalize images for DINOv2 (expects ImageNet normalization)"""
        # Input img shape: (B, C, H, W) with values in [0, 255] from ImageTokenizerPt
        # DINOv2 expects: (B, C, H, W) with ImageNet normalization
        
        # Check if already in correct format (B, C, H, W) where C is small (3 or 6)
        if img.shape[1] not in [3, 6]:  # If channels not in second position
            # Assume it's (B, H, W, C) and convert to (B, C, H, W)
            img = img.permute(0, 3, 1, 2)
        
        # Normalize to [0, 1]
        img = img.float() / 255.0
        if self.img_norm_type == "imagenet":
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(img.device)
            img = (img - mean) / (std)
        
        # Resize to 224x224 if not already (DINOv2 default size)
        if img.shape[2] != 224 or img.shape[3] != 224:
            img = nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        
        return img
    
    def forward(self, observations: torch.Tensor, train: bool = True, **kwargs):
        """
        Args:
            observations: (B, C, H, W) image tensor with values in [0, 255]
                         C can be 3 (single image) or 6 (stacked images)
        
        Returns:
            image_tokens: (B, embed_dim, H_patches, W_patches) tensor
                         For 224x224 input with patch_size=14: (B, 384, 16, 16)
        """
        # Normalize images (handles resizing to 224x224 internally)
        x = self.normalize_images(observations)
        
        # Set model to eval mode if frozen
        if self.freeze:
            self.dino_model.eval()
        
        # Forward through DINO (no gradients if frozen)
        with torch.set_grad_enabled(not self.freeze and train):
            # Get patch embeddings (without CLS token)
            features = self.dino_model.forward_features(x)
            
            # features shape: (B, num_patches + 1, embed_dim)
            # Remove CLS token (first token)
            patch_tokens = features['x_norm_patchtokens']  # (B, num_patches, embed_dim)
        
        # Convert from (B, num_patches, embed_dim) to (B, embed_dim, H_patches, W_patches)
        # This matches the format expected by ImageTokenizerPt: (B, C, H, W)
        # For 224x224 image with patch_size=14: num_patches = 16*16 = 256
        B = patch_tokens.shape[0]
        num_patches = patch_tokens.shape[1]
        embed_dim = patch_tokens.shape[2]
        
        # Calculate grid size
        grid_size = int(num_patches ** 0.5)
        
        # Reshape to spatial grid and permute to (B, C, H, W) format
        patch_tokens = patch_tokens.reshape(B, grid_size, grid_size, embed_dim)
        patch_tokens = patch_tokens.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        return patch_tokens


class DinoV3EncoderPt(nn.Module):
    """DINOv3 Vision Encoder loaded from a local checkpoint via torch.hub.

    Supported model names (patch_size=16):
        dinov3_vits16  → embed_dim=384
        dinov3_vitb16  → embed_dim=768
        dinov3_vitl16  → embed_dim=1024
        dinov3_vitg16  → embed_dim=1536
    """
    
    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        freeze: bool = True,
        img_norm_type: str = "imagenet",
        dinov3_path: Optional[str] = None,
        ckpt_root: Optional[str] = None,
    ):
        # DINOv3 weights are gated, so they are loaded from a local clone of
        # https://github.com/facebookresearch/dinov3 plus a downloaded .pth file.
        # Paths default to the DINOV3_REPO / DINOV3_CKPT_ROOT environment variables.
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.img_norm_type = img_norm_type

        self.ckpt_root = ckpt_root or os.environ.get("DINOV3_CKPT_ROOT", "./pretrained")
        self.dinov3_path = dinov3_path or os.environ.get("DINOV3_REPO", "./dinov3")

        self._DINOV3_CKPTS = {
            "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        }
        
        ckpt = os.path.join(self.ckpt_root, self._DINOV3_CKPTS[model_name])

        self.dino_model = torch.hub.load(
            self.dinov3_path,
            model_name,
            source="local",
            weights=ckpt,
        )

        self.num_features = self.dino_model.num_features
        self.patch_size = self.dino_model.patch_size  # 16

        if self.freeze:
            for param in self.dino_model.parameters():
                param.requires_grad = False
            self.dino_model.eval()

    def normalize_images(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] not in [3, 6]:
            img = img.permute(0, 3, 1, 2)
        if img.max() > 1.0:
            img = img.float() / 255.0
        if self.img_norm_type == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(img.device)
            img = (img - mean) / std
        target = (256 // self.patch_size) * self.patch_size  # nearest multiple of patch_size
        if img.shape[2] != target or img.shape[3] != target:
            img = nn.functional.interpolate(img, size=(target, target), mode="bilinear", align_corners=False)
        return img

    def forward(self, observations: torch.Tensor, train: bool = True, **kwargs) -> torch.Tensor:
        """
        Args:
            observations: (B, C, H, W) uint8 image tensor.
        Returns:
            patch_tokens: (B, embed_dim, H_patches, W_patches)
        """
        x = self.normalize_images(observations)
        if self.freeze:
            self.dino_model.eval()
        with torch.set_grad_enabled(not self.freeze and train):
            features = self.dino_model.forward_features(x)
            patch_tokens = features["x_norm_patchtokens"]  # (B, num_patches, D)

        B, num_patches, embed_dim = patch_tokens.shape
        grid_size = int(num_patches ** 0.5)
        patch_tokens = patch_tokens.reshape(B, grid_size, grid_size, embed_dim)
        patch_tokens = patch_tokens.permute(0, 3, 1, 2)  # (B, D, H, W)
        return patch_tokens
