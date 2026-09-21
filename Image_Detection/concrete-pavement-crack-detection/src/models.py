"""
Model definitions and transfer learning architectures for Concrete Crack Detection.
Supports ResNet-18, MobileNetV2, and MobileNetV3-Small with customizable heads.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn
from torchvision import models


class ConcreteCrackClassifier(nn.Module):
    """
    Modular transfer learning wrapper for binary concrete crack classification.
    Outputs raw logits suitable for BCEWithLogitsLoss (numerically stable).
    """
    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.dropout_rate = dropout

        if self.model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.base = models.resnet18(weights=weights)
            in_features = self.base.fc.in_features
            # Replace final FC layer
            self.base.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1)
            )

        elif self.model_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.base = models.mobilenet_v2(weights=weights)
            in_features = self.base.classifier[1].in_features
            # Replace classifier
            self.base.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1)
            )

        elif self.model_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.base = models.mobilenet_v3_small(weights=weights)
            in_features = self.base.classifier[0].in_features
            self.base.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.Hardswish(),
                nn.Dropout(p=dropout),
                nn.Linear(512, 1)
            )

        elif self.model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.base = models.resnet50(weights=weights)
            in_features = self.base.fc.in_features
            self.base.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1)
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'resnet18', 'mobilenet_v2', 'mobilenet_v3_small', or 'resnet50'.")

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits of shape (BatchSize, 1)."""
        return self.base(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns crack probability in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def freeze_backbone(self) -> None:
        """Freezes all backbone feature extraction parameters."""
        for name, param in self.base.named_parameters():
            # Keep classifier / fc trainable
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreezes all parameters for full fine-tuning."""
        for param in self.base.parameters():
            param.requires_grad = True

    def unfreeze_top_layers(self, num_blocks: int = 2) -> None:
        """Unfreezes the top convolutional blocks for fine-tuning."""
        if "resnet" in self.model_name:
            # ResNet blocks: layer1, layer2, layer3, layer4
            layers_to_unfreeze = [f"layer{4 - i}" for i in range(min(num_blocks, 4))]
            for name, child in self.base.named_children():
                if name in layers_to_unfreeze or name == "fc":
                    for param in child.parameters():
                        param.requires_grad = True
        elif self.model_name == "mobilenet_v2":
            # MobileNetV2 inverted residual blocks in features
            total_features = len(self.base.features)
            cutoff = max(0, total_features - (num_blocks * 2))
            for idx, child in enumerate(self.base.features):
                if idx >= cutoff:
                    for param in child.parameters():
                        param.requires_grad = True

    def get_target_layer_for_cam(self) -> nn.Module:
        """Returns the final convolutional layer for Grad-CAM activation mapping."""
        if "resnet" in self.model_name:
            return self.base.layer4[-1]
        elif self.model_name == "mobilenet_v2":
            return self.base.features[-1]
        elif self.model_name == "mobilenet_v3_small":
            return self.base.features[-1]
        raise NotImplementedError(f"Target layer not configured for {self.model_name}")


def build_model(
    model_name: str = "resnet18",
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False
) -> ConcreteCrackClassifier:
    """Helper factory function to create crack classification model."""
    return ConcreteCrackClassifier(
        model_name=model_name,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )


if __name__ == "__main__":
    print("Testing model creation and forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    dummy_input = torch.randn(2, 3, 224, 224, device=device)

    for arch in ["resnet18", "mobilenet_v2"]:
        model = build_model(arch, pretrained=True, freeze_backbone=True).to(device)
        output = model(dummy_input)
        probs = model.predict_proba(dummy_input)
        cam_layer = model.get_target_layer_for_cam()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"[{arch.upper()}] Forward output shape: {output.shape}, Probs: {probs.squeeze().tolist()}")
        print(f"  Total params: {total_params:,}, Trainable (frozen backbone): {trainable_params:,}")
        print(f"  Grad-CAM layer identified: {type(cam_layer).__name__}")
