import torch
import logging


MODELS_DIR = "checkpoints"


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return device

def map_model_to_resnet50(checkpoint_model):
    from src.models.resnet import ResNet50
    if isinstance(checkpoint_model, ResNet50):
        return checkpoint_model
    model = ResNet50()
    state_dict = checkpoint_model.state_dict()
    formatted_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    model.load_state_dict(formatted_state_dict, strict=False)
    model = model.to(get_device())
    return model
