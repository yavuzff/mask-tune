import torch.nn as nn
import timm

class ResNet50(nn.Module):
    """
    Standard ResNet50 architecture using pretrained ImageNet weights.
    """
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        # use timm to easily load the model and swap the classification head
        self.model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)

    def get_cam_target_layers(self):
        """
        Target the final bottleneck block in layer 4 for Grad-CAM.
        """
        return [self.model.layer4[-1]]

    def forward(self, x):
        return self.model(x)
