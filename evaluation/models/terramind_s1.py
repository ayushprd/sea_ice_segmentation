"""TerraMind S1 model for SAR sea ice segmentation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY


class SegmentationHead(nn.Module):
    """Simple segmentation head: upsample + conv layers."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        x = self.head(x)
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        return x


class TerraMindS1Segmentation(nn.Module):
    """TerraMind encoder with S1 SAR input for sea ice segmentation.

    Uses TerraMind-1.0 backbone with Sentinel-1 GRD modality.

    Note: TerraMind was trained on VV/VH polarization, but AI4Arctic
    provides HH/HV. We treat them as equivalent for transfer learning.
    """

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        # TerraMind S1 configuration
        # TerraMind expects 2 bands: VV and VH (we'll use HH, HV as proxy)
        self.band_names = ['VV', 'VH']  # Use these names for TerraMind compatibility
        self.modality = 'untok_sen1grd@224'

        # Normalization for TerraMind S1
        # TerraMind expects dB values, typical range for sea ice SAR
        self.register_buffer('mean', torch.tensor([
            -12.54, -20.33  # VV, VH typical values (dB)
        ]).view(1, 2, 1, 1))
        self.register_buffer('std', torch.tensor([
            5.25, 5.42
        ]).view(1, 2, 1, 1))

        # Load backbone
        self.backbone = TERRATORCH_BACKBONE_REGISTRY.build(
            'terramind_v1_base',
            pretrained=True,
            modalities=[self.modality],
            bands={self.modality: self.band_names},
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # TerraMind embedding dimension
        self.embedding_dim = 768

        # Segmentation head
        self.head = SegmentationHead(
            in_channels=self.embedding_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 2, H, W) S1 SAR input (HH, HV or VV, VH polarization)
               Assumed to be in dB-like scale

        Returns:
            (B, num_classes, H, W) logits
        """
        B, C, H, W = x.shape

        # Normalize
        x = (x - self.mean) / (self.std + 1e-6)

        # Extract features
        features = self.backbone(x)
        last_features = features[-1]

        # Handle different output formats
        # TerraMind outputs (B, 196, 768) - 14x14 spatial tokens
        if last_features.dim() == 3:
            n_tokens = last_features.shape[1]
            h = w = int(math.sqrt(n_tokens))
            spatial_features = last_features.transpose(1, 2).reshape(B, -1, h, w)
        else:
            spatial_features = last_features

        # Decode
        logits = self.head(spatial_features, (H, W))
        return logits


class TerraMindS1S2Segmentation(nn.Module):
    """TerraMind with both S1 (SAR) and S2 (optical) for multimodal sea ice segmentation.

    Note: This requires both S1 and S2 data, which may not always be available
    for Arctic regions due to polar night.
    """

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        # S1 configuration
        self.s1_bands = ['VV', 'VH']
        self.s1_modality = 'untok_sen1grd@224'

        # S2 configuration (12 bands)
        self.s2_bands = [
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06',
            'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'
        ]
        self.s2_modality = 'untok_sen2l2a@224'

        # Load backbone with both modalities
        self.backbone = TERRATORCH_BACKBONE_REGISTRY.build(
            'terramind_v1_base',
            pretrained=True,
            modalities=[self.s1_modality, self.s2_modality],
            bands={
                self.s1_modality: self.s1_bands,
                self.s2_modality: self.s2_bands,
            },
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.embedding_dim = 768

        self.head = SegmentationHead(
            in_channels=self.embedding_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """Forward pass with multimodal input.

        Args:
            s1: (B, 2, H, W) S1 SAR input
            s2: (B, 12, H, W) S2 optical input

        Returns:
            (B, num_classes, H, W) logits
        """
        B, _, H, W = s1.shape

        # Create multimodal input dict
        x = {
            self.s1_modality: s1,
            self.s2_modality: s2,
        }

        # Extract features
        features = self.backbone(x)
        last_features = features[-1]

        # Handle output format
        if last_features.dim() == 3:
            n_tokens = last_features.shape[1]
            h = w = int(math.sqrt(n_tokens))
            spatial_features = last_features.transpose(1, 2).reshape(B, -1, h, w)
        else:
            spatial_features = last_features

        logits = self.head(spatial_features, (H, W))
        return logits
