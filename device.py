import torch


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"  # nvidia
    if torch.backends.mps.is_available():
        return "mps"   # apple neural engine
    return "cpu"