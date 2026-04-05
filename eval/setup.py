import os
import torch
import numpy as np
import random
import logging
import warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Required for deterministic cuBLAS ops; must be set before CUDA init,
    # but setting it here covers cases where CUDA hasn't been used yet.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # warn_only=True avoids hard errors from bitsandbytes ops that lack
    # a deterministic kernel implementation.
    torch.use_deterministic_algorithms(True, warn_only=True)

def configure_logging():
    logging.basicConfig(level=logging.ERROR)
