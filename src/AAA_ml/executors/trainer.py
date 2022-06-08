"""Training executors
"""
import logging
from pathlib import Path
from shutil import copyfile

import confuse
import numpy as np
import talos as ta
# local package
from tf_mmciad.utils.hyper import TalosModel, single_run

from utils.utils import Parameters, predefined_datagenerator


def execute_grid_search(
    config: confuse.AttrDict,
    rt: confuse.AttrDict,
    paths: confuse.AttrDict,
    params: Parameters,
    tb_callback_params: dict,
    logger: logging.Logger,
):
    """Execute a grid search on the training data .

    Returns:
        [TalosModel]: [description]
    """
    logger.info("Running 'execute_grid_search' branch ...")
    train_generator = predefined_datagenerator(
        paths.train, params=params, id_list=params.train_ids
    )
    val_generator = predefined_datagenerator(
        paths.train, params=params, augment=False, id_list=params.val_ids
    )

    statics = config["statics"]
    statics["date"] = rt.date_string
    grid_params = config["grid_params"]

    talos_model = TalosModel(
        paths.models,
        statics,
        train_generator,
        val_generator,
        params.cls_wgts,
        tb_callback_params,
        debug=rt.debug,
    )

    if not (dest := Path(paths.models, rt.date_string, "config.yaml")).exists():
        copyfile("config.yaml", dest)

    dummy_x = np.empty((1, params.batch_size, *params.dim))
    dummy_y = np.empty((1, params.batch_size))

    scan_obj = ta.Scan(
        x=dummy_x,
        y=dummy_y,
        disable_progress_bar=False,
        print_params=True,
        model=talos_model,
        params=grid_params,
        experiment_name=paths.grid + rt.date_string,
        # reduction_method='gamify',
        allow_resume=True,
    )
    return scan_obj


def execute_single_train(
    config: confuse.AttrDict,
    rt: confuse.AttrDict,
    paths: confuse.AttrDict,
    params: confuse.AttrDict,
    tb_callback_params: dict,
    logger: logging.Logger,
    error_resume: bool,
):
    """Execute a single train and return a model and history .

    Returns:
        [type]: [description]
    """
    logger.info("Running 'execute_single_train' branch ...")
    train_generator = predefined_datagenerator(
        paths.train, params=params, id_list=params.train_ids
    )
    val_generator = predefined_datagenerator(
        paths.train, params=params, augment=False, id_list=params.val_ids
    )
    statics = config["statics"]
    statics["date"] = rt.date_string
    train_model, history = single_run(
        paths.models,
        statics,
        train_generator,
        val_generator,
        cls_wgts=params.cls_wgts,
        debug=rt.debug,
        tensorboard_params=tb_callback_params,
        resume=False,
        resume_on_error=error_resume,
    )
    if not (dest := Path(paths.models, rt.date_string, "config.yaml")).exists():
        copyfile("config.yaml", dest)
    return train_model, history
