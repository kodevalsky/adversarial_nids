import random
import numpy as np
import torch

def set_deterministic_seed(seed):
    """Locks all random seeds for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)