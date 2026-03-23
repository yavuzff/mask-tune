import torch.nn as nn
import timm

class StandardViT(nn.Module):
    """
    Standard ViT-Small (Patch 16, 224x224) using pretrained ImageNet weights.
    """
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        # vit_small is the closest equivalent to ResNet-50 in the ViT family
        self.model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes)

    def get_cam_target_layers(self):
        """
        For ViTs, we target the LayerNorm of the final Transformer block.
        """
        return [self.model.blocks[-1].norm1]

    def forward(self, x):
        return self.model(x)


class TinyViTMNIST(nn.Module):
    """
    Tiny ViT shaped specifically for 28x28 MNIST images.
    Uses 4x4 patches to create a 7x7 spatial token grid.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # pretrained=False so we would need to retrain
        self.model = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            num_classes=num_classes,
            img_size=28,
            patch_size=4,
            in_chans=3,
            embed_dim=8,  # reduced from default 192
            depth=4,  # reduced from default 12
            num_heads=2  # reduced from default 3
        )
        # print number of parameters
        print(f"Tiny ViT has {sum(p.numel() for p in self.model.parameters())} parameters")

    def get_cam_target_layers(self):
        return [self.model.blocks[-1].norm1]

    def forward(self, x):
        return self.model(x)
