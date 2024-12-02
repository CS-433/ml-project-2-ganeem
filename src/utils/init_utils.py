import random
import numpy as np
import torch
from pathlib import Path


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_project_root() -> Path:
    """
    Get the path to the project root directory.

    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent


ROOT_PATH = get_project_root()
