"""Misc utils
"""
import fileinput
import re
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Optional, Union

import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
from tf_mmciad.model.layers import MyConv2D
from tf_mmciad.utils.custom_loss import (categorical_focal_loss, jaccard1_coef,
                                         jaccard1_loss, jaccard2_coef,
                                         jaccard2_loss, tversky_loss)
from tf_mmciad.utils.f_scores import F1Score
# local package
from tf_mmciad.utils.generator import DataGenerator
from tf_mmciad.utils.swish import Swish


@dataclass
class Parameters:
    """Set parameters for training .
    """

    x_min: int = 1
    x_max: int = 255
    mean: Union[List[int], np.ndarray] = None
    stdev: Union[List[int], np.ndarray] = None
    colors: Dict[int, List[int]] = None
    colorvec: np.ndarray = None
    batch_size: int = 16
    dim: List[int] = None
    n_channels: int = 3
    n_classes: int = 5
    class_names: List[str] = None
    cls_wgts: Dict[int, float] = None
    train_ids: List[str] = None
    val_ids: List[str] = None


@dataclass
class CustomObjects:
    """Class customObjects to be used for loading previously trained models.
    """

    layers = {
        "MyConv2D": MyConv2D,
        "Swish": Swish,
    }
    losses = {
        "cat_CE": categorical_crossentropy,
        "tversky_loss": tversky_loss,
        "categorical_focal_loss_fixed": categorical_focal_loss(),
        "cat_FL": categorical_focal_loss(),
        "jaccard1_loss": jaccard1_loss,
        "jaccard2_loss": jaccard2_loss,
    }
    metrics = {
        "jaccard1_coef": jaccard1_coef,
        "jaccard2_coef": jaccard2_coef,
        "F1Score": F1Score,
    }

    def get(self) -> dict:
        """Get a dict containing the custom metrics, losses and layers specified above.

        Returns:
            dict
        """
        return {**self.metrics, **self.losses, **self.layers}


def get_filename(path: Union[str, Path]) -> str:
    """Return the basename stripped of extension
    """
    return Path(path).stem


def file_counter(path: Path) -> int:
    """Returns the number of files in a directory .

    Args:
        path (Path): [description]

    Returns:
        int: [description]
    """
    if path.exists():
        return len([name for name in path.iterdir() if name.is_file()])
    return 0


def rewrite_config_flag(flag: str = "clean_start", state: bool = False):
    """Rewrite the config.yaml file with the given flag and state.

    Args:
        flag (str, optional): name of the config flag. Defaults to "clean_start".
        state (bool, optional): flag state to be written. Defaults to False.
    """
    copyfile("config.yaml", "config.yaml.backup")
    with fileinput.FileInput("config.yaml", inplace=True) as inputfile:
        for line in inputfile:
            print(
                re.sub(
                    r"(?<=" + flag + r": )(true|false)",
                    str(state).lower(),
                    line.rstrip(),
                )
            )


def write_to_config(subs: Optional[Dict[str, float]] = None):
    """Writes a dictionary of configuration values to a config.yaml file and sets

    Args:
        subs (Dict[str, float]): key-value pairs for data set statistics to be written
            to config
    """
    rewrite_config_flag("clean_start", False)
    if subs:
        for key, val in subs.items():
            pattern = re.compile(r"(?<=" + key + r":)([ ]+)([\[0-9\., \-\]]+)(?= #)")
            with fileinput.FileInput("config.yaml", inplace=True) as inputfile:
                for line in inputfile:
                    line = re.sub(
                        pattern,
                        r"\g<1>"
                        + np.array2string(val, separator=", ").replace("\n", ""),
                        line.rstrip(),
                    )
                    print(line)


def npa2s(arr, *args, **kwargs):
    """Return a string representation of an array.
    """
    return np.array2string(arr, *args, **kwargs)


def predefined_datagenerator(
    path: str, params: Parameters, augment: bool = True, **kwargs
) -> DataGenerator:
    """Returns a predefined DataGenerator based on parameters from the config file.

    Args:
        path (str): [description]
        augment (bool, optional): [description]. Defaults to True.

    Returns:
        DataGenerator: [description]
    """
    return DataGenerator(
        path,
        params.colors,
        params.mean,
        params.stdev,
        params.x_min,
        params.x_max,
        batch_size=params.batch_size,
        dim=params.dim,
        n_channels=params.n_channels,
        n_classes=params.n_classes,
        shuffle=True,
        augment=augment,
        **kwargs,
    )


def write_results(
    path: Union[str, Path],
    rank: str,
    scores: dict,
    mode: str = "w",
):
    """Write the results to a file.

    Args:
        path (Union[str, Path]): [description]
        rank (str): [description]
        scores (dict): [description]
        mode (str, optional): [description]. Defaults to "w".
    """
    with open(path, mode=mode) as res_file:
        res_file.write(f"Model: {rank}\n")
        for param, value in scores.items():
            if isinstance(value, (str, np.ndarray)):
                res_file.write(f"{param}:\n{value}\n")
            else:
                res_file.write(f"{param} = {value}\n")
        res_file.write("---------------------------------------------\n\n")
