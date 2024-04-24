import torch


def model_size(model: torch.nn.Module) -> int:
    return sum([p.numel() for p in model.parameters()])

