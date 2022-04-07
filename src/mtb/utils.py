import random
from typing import Dict, Any

import torch
import numpy as np
from omegaconf import DictConfig
from hydra.utils import to_absolute_path


# names of hyper-parameters
_HYPERPARAM_NAMES = [
    "batch_size",
    "hidden_size",
    "latent_size",
    "num_epochs",
    "weight_decay",
]


def seed_everything(seed: int) -> None:
    """Sets random seed anywhere randomness is involved.
    This process makes sure all the randomness-involved operations yield the
    same result under the same `seed`, so each experiment is reproducible.
    In this function, we set the same random seed for the following modules:
    `random`, `numpy` and `torch`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_relative_path(cfg: DictConfig) -> None:
    """Resolves all the relative path(s) given in `config.dataset` into absolute path(s).
    This function makes our code runnable in docker as well, where using relative path has
    problem with locating dataset files in `src/../datasets`.

    Args:
        cfg: Configuration of the experiment given in a dict.

    Example:
        Given `cfg.dataset.root="./datasets` and we call from
        "/netscratch/user/code/mtb/main.py", then `cfg.dataset.root` is
        overwritten by `/netscratch/user/code/mbt/datasets`.
    """
    for file_path in ["train_file", "eval_file"]:
        cfg[file_path] = to_absolute_path(cfg[file_path])


def read_hyperparams_from_cfg(cfg: DictConfig) -> Dict[str, Any]:
    """Read hyperparameters from configuration.

    Args:
        cfg: Configuration of the experiment given in a dict.

    Returns:
        A dictionary containing the hyperparameters from `cfg`.
    """
    return {param_name: cfg[param_name] for param_name in _HYPERPARAM_NAMES}
