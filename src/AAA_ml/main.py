"""AAA Image Segmentation
This script will set up and either train a network or predict segmentations based on
previous training
"""
import logging
import platform
import re
import sys
from datetime import date
from pathlib import Path

import confuse
import numpy as np
import silence_tensorflow
import tensorflow as tf

# local package

from tf_mmciad.utils.custom_loss import (
    tversky_loss,
    weighted_loss,
)

from tf_mmciad.utils.io import (
    create_samples,
)
from tf_mmciad.utils.preprocessing import (
    calculate_class_weights,
    calculate_stats,
    class_ratio,
)
from executors.trainer import execute_grid_search, execute_single_train
from executors.predicter import execute_predict
from executors.analyzer import execute_analysis
from executors.evaluator import execute_evaluation
from utils.utils import (Parameters, CustomObjects, file_counter,
                          rewrite_config_flag, write_to_config, npa2s)


def main():
    """Runs the main program and prints the output
    
    Currently, no arguments can be supplied at command line,
    but the config file can be edited to suit your needs.
    """

    silence_tensorflow.silence_tensorflow()  # Stop all unnecessary TF prints to stderr
    print(
        f"Running Python version {platform.python_version()}",
        file=sys.stderr,
        flush=True,
    )
    print(f"Build {sys.version}", file=sys.stderr, flush=True)
    print(f"TensorFlow version {tf.__version__}", file=sys.stderr, flush=True)

    ###################################
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print("Num GPUs:", len(gpus), file=sys.stderr, flush=True)
    # TensorFlow wizardry
    # tf.debugging.set_log_device_placement(True)

    # Don't pre-allocate memory; allocate as-needed

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus),
                "Physical GPUs,",
                len(logical_gpus),
                "Logical GPUs",
                file=sys.stderr,
                flush=True,
            )
        except RuntimeError as err:
            # Memory growth must be set before GPUs have been initialized
            print(err)

    # Only allow a total of half the GPU memory to be allocated
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5

    ###################################

    template = {
        "appName": str,
        "Runtime": {
            "clean_start": bool,
            "compute_stats": bool,
            "compute_class_weights": bool,
            "date_string": str,
            "debug": bool,
            "make_abundance_table": bool,
            "run_grid_search": bool,
            "run_train": bool,
            "run_predict": bool,
            "run_eval": bool,
            "analyze": bool,
            "create_probability_maps": bool,
        },
        "data": confuse.MappingValues(confuse.OneOf([str, confuse.Sequence([str])])),
        "class_map": confuse.MappingValues(int),
        "class_colors": confuse.MappingValues(confuse.Sequence([int])),
        "active_labels": confuse.Sequence([int]),
        "ignore_cls": confuse.Optional(int, default=None),
        "stats": confuse.MappingValues(confuse.Sequence([float])),
        "input_meta": {
            "x": int,
            "y": int,
            "channels": int,
            "tile_size": confuse.Sequence([int]),
            "shape": confuse.Sequence([confuse.Optional(int, default=None)]),
        },
        "batch_size": int,
        "nb_epoch": int,
        "nb_frozen": int,
        "verbose": int,
        "drop": int,
        "statics": {
            "shape": confuse.Sequence([confuse.Optional(int, default=None)]),
            "nb_epoch": int,
            "nb_filters_0": int,
            "batch_size": int,
            "verbose": int,
            "num_cls": int,
            "batchnorm": bool,
            "maxpool": bool,
            "opt": str,
            "depth": int,
            "arch": str,
            "dropout": float,
            "decay": float,
            "sigma_noise": float,
            "act": str,
            "pretrain": int,
            "lr": float,
            "class_weights": bool,
            "loss_func": str,
            "init": str,
        },
        "grid_params": {
            "nb_filters_0": confuse.Sequence([int]),
            "depth": confuse.Sequence([int]),
            "loss_func": confuse.Sequence([str]),
            "arch": confuse.Sequence([str]),
            "act": confuse.Sequence([str]),
            "opt": confuse.Sequence([str]),
            "init": confuse.Sequence([str]),
        },
    }

    config = confuse.Configuration("AAAml", __name__)
    config.set_file("/nb_projects/AAA_ml/config.yaml")
    config = config.get(template)

    # if date_string is None, current date will be used
    rt: confuse.AttrDict = config.Runtime

    class_map = config.class_map

    class_colors = config.class_colors

    active_labels = config.active_labels
    active_classes = [sorted(class_map, key=class_map.get)[i] for i in active_labels]
    print(active_classes, file=sys.stderr, flush=True)
    colorvec = np.asarray([class_colors[i] for i in active_labels])
    active_colors = {class_: class_colors[class_] for class_ in active_labels}
    print(active_colors, file=sys.stderr, flush=True)
    print(colorvec.shape, file=sys.stderr, flush=True)
    ignore_cls = config.ignore_cls  # change back to 0 if necessary

    paths = confuse.AttrDict(config.data)
    # "models"
    target_ext: str = paths.target_ext
    patient_val_ids: list = paths.val_list

    tile_size = (*config.input_meta.tile_size,)

    logger = logging.getLogger(__name__)
    c_handler = logging.StreamHandler(sys.stderr)
    f_path = Path(".", "logs", rt.date_string)
    f_path.mkdir(parents=True, exist_ok=True)
    if rt.debug:
        logger.setLevel(logging.DEBUG)
        f_handler = logging.FileHandler(f_path / "debug.log")
        f_handler.setLevel(logging.DEBUG)
        c_handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        f_handler = logging.FileHandler(f_path / "error.log")
        f_handler.setLevel(logging.INFO)
        c_handler.setLevel(logging.INFO)
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
    )
    f_handler.setFormatter(log_format)
    c_handler.setFormatter(log_format)
    logger.addHandler(f_handler)
    logger.addHandler(c_handler)

    if rt.clean_start:
        # duplicate = [class_colors[label] for label in [2, 3, 6, 9, 11]]

        filter_dict = {
            "None": (class_map["None"], 0.8, 0.9, None),
            "Ignore*": (class_map["Ignore*"], 0.8, 0.9, None),
        }
        for path in Path(paths.train).glob("**/*." + target_ext):
            path.unlink()
        create_samples(
            Path(paths.wsi),
            filter_dict,
            prefix="",
            output_dir="../training",
            tile_size=tile_size,
            # duplication_list=class_map["Thrombus"],
        )
        rewrite_config_flag("clean_start", False)

    # Denne liste skal indeholde alle tiles der matcher ID'et fra patient_val_ids
    # bsf burde blive defineret dynamisk.
    # En tilsvarende liste skal genereres for val_ids
    tiles = sorted(Path(paths.train).glob("*." + target_ext))
    train_ids = [
        tile.stem
        for tile in tiles
        if not re.match(r"(" + "|".join(patient_val_ids) + r")_[0-9]+", tile.stem)
    ]
    val_ids = [
        tile.stem
        for tile in tiles
        if re.match(r"(" + "|".join(patient_val_ids) + r")_[0-9]+", tile.stem)
    ]
    logger.debug(f"{tiles[0] = }\n{train_ids[:5] = }")

    if len(train_ids) + len(val_ids) != len(tiles):
        raise ValueError(
            "Wrong number of tiles in lists. "
            + f"{len(train_ids)=}\n{len(val_ids)=}\n{len(tiles)=}"
        )

    path_dict = {
        "x": Path(paths.train),
        "y": Path(paths.train, "gt"),
        "xv": Path(paths.val),
        "yv": Path(paths.val, "gt"),
        "xt": Path(paths.test),
        "yt": Path(paths.test, "gt"),
    }

    print(
        """
        Train set:
        ---------
        Created {} X tiles in directory: {}.
        Created {} Y tiles in directory: {}.

        Validation set:
        ---------------
        Created {} X tiles in directory: {}.
        Created {} Y tiles in directory: {}.

        Test set:
        ---------
        Created {} X tiles in directory {}.
        Created {} Y tiles in directory {}.""".format(
            file_counter(path_dict["x"]),
            path_dict["x"],
            file_counter(path_dict["y"]),
            path_dict["y"],
            file_counter(path_dict["xv"]),
            path_dict["xv"],
            file_counter(path_dict["yv"]),
            path_dict["yv"],
            file_counter(path_dict["xt"]),
            path_dict["xt"],
            file_counter(path_dict["yt"]),
            path_dict["yt"],
        ),
        file=sys.stderr,
        flush=True,
    )

    if not rt.analyze and (rt.clean_start or rt.compute_stats):
        train_m, train_s, x_min, x_max = calculate_stats(
            path="data/WSI/", prefix=""
        )  # , local=False)
    else:
        train_m = np.array(config["stats"]["train_m"])
        train_s = np.array(config["stats"]["train_s"])
        x_min = np.array(config["stats"]["x_min"])
        x_max = np.array(config["stats"]["x_max"])

    print(
        f"Dataset\n-------\nMean:    {npa2s(train_m, separator=', '):}\n",
        f"St.dev.: {npa2s(train_s, separator=', '):}",
    )
    print(
        f"Min:     {npa2s(x_min, separator=', ')}\n",
        f"Max:     {npa2s(x_max, separator=', ')}",
    )

    if not rt.analyze and (rt.clean_start or rt.compute_stats):
        stats = {"train_m": train_m, "train_s": train_s, "x_min": x_min, "x_max": x_max}
        write_to_config(stats)

    if rt.compute_class_weights:
        # Calculate the class weights for the training data set
        # Optionally exclude a label by settings its weight to 0 using the
        # ignore=label option
        cls_wgts = calculate_class_weights(
            paths.data, active_labels, class_colors, ignore=ignore_cls
        )
        # TODO write to config
        print("cls_wgts:\n", cls_wgts)
        class_ratios = class_ratio(paths.data, active_labels, class_colors)
        print("class_ratios:\n", class_ratios)
    else:
        # Replace with a read from config later
        if isinstance(ignore_cls, int):
            cls_wgts = {i: 1 if i != ignore_cls else 0 for i in active_labels}
        else:
            cls_wgts = None

    if rt.compute_class_weights:
        w_tversky_loss = weighted_loss(tversky_loss, cls_wgts)
        w_tversky_loss.__name__ = "w_TL"

    # architecture params

    # ****  deep learning model
    batch_size = config.batch_size
    inference_tile_size = (1024, 1024)

    tb_callback_params = {
        "path": paths.test,
        "color_dict": active_colors,
        "color_list": colorvec,
        "means": train_m,
        "stds": train_s,
        "x_min": x_min,
        "x_max": x_max,
        "batch_size": 7,
        "dim": tile_size,
        "n_channels": config.input_meta.channels,
        "n_classes": config.statics.num_cls,
    }

    train_params = Parameters(
        x_min=x_min,
        x_max=x_max,
        mean=train_m,
        stdev=train_s,
        colors=active_colors,
        batch_size=batch_size,
        dim=tile_size,
        n_channels=config.input_meta.channels,
        n_classes=config.statics.num_cls,
        cls_wgts=cls_wgts,
        train_ids=train_ids,
        val_ids=val_ids,
    )

    pred_params = Parameters(
        x_min=x_min,
        x_max=x_max,
        mean=train_m,
        stdev=train_s,
        colors=active_colors,
        colorvec=colorvec,
        batch_size=batch_size,
        dim=inference_tile_size,
        n_channels=config.input_meta.channels,
        n_classes=config.statics.num_cls,
        class_names=active_classes,
        cls_wgts=cls_wgts,
        train_ids=train_ids,
        val_ids=val_ids,
    )

    eval_params = Parameters(
        x_min=x_min,
        x_max=x_max,
        mean=train_m,
        stdev=train_s,
        colors=active_colors,
        batch_size=batch_size,
        dim=tile_size,
        n_channels=config.input_meta.channels,
        n_classes=config.statics.num_cls,
        cls_wgts=cls_wgts,
        train_ids=train_ids,
        val_ids=val_ids,
    )

    analyse_params = Parameters(
        x_min=x_min,
        x_max=x_max,
        colors=active_colors,
        colorvec=colorvec,
        batch_size=batch_size,
        dim=inference_tile_size,
        n_channels=config.input_meta.channels,
        n_classes=config.statics.num_cls,
        class_names=active_classes,
        cls_wgts=cls_wgts,
        train_ids=train_ids,
        val_ids=val_ids,
    )

    custom_objects = CustomObjects()

    ###################################################################################

    if rt.run_grid_search:
        scan_object = execute_grid_search(
            config, rt, paths, train_params, tb_callback_params, logger
        )

    if rt.run_train:
        if not [
            f
            for f in Path(paths.models, rt.date_string).iterdir()
            if f.is_dir() and "model" in str(f)
        ]:
            _ = execute_single_train(
                config, rt, paths, train_params, tb_callback_params, logger, False
            )
        elif not Path(paths.models, rt.date_string, "saved_model").exists():
            logger.info("Saved_model does not exist. Retrying...")
            _ = execute_single_train(
                config, rt, paths, train_params, tb_callback_params, logger, True
            )
        else:
            logger.info("Training previously completed!")

    if rt.run_predict:
        # TODO Add command line parameter to choose files
        if not rt.date_string:
            rt.date_string = str(date.today())

        if not rt.run_grid_search and rt.run_train:
            _ = execute_predict(
                paths,
                rt,
                patient_val_ids,
                pred_params,
                custom_objects,
                rt.date_string,
                logger,
            )
        elif rt.run_grid_search and not rt.run_train:
            try:
                scan_object
            except NameError:
                scan_object = execute_grid_search(
                    config, rt, paths, train_params, tb_callback_params, logger
                )
            finally:
                _ = execute_predict(
                    paths,
                    rt,
                    patient_val_ids,
                    pred_params,
                    custom_objects,
                    rt.date_string,
                    logger,
                    scan_object=scan_object,
                )
        else:
            raise ValueError(
                "Config file misconfigured!\n\t"
                + "Either 'run_grid_search' or 'rt.run_train' should be True"
            )

    if rt.run_eval:
        if not rt.date_string:
            rt.date_string = str(date.today())
        if not rt.run_grid_search and rt.run_train:
            evaluation = execute_evaluation(
                paths, eval_params, custom_objects, rt.date_string, logger,
            )
            with open(
                Path(paths.results, rt.date_string, "eval.txt"), mode="w"
            ) as res_file:
                for param, value in evaluation.items():
                    res_file.write(f"{param} = {value}\n")
        else:
            raise ValueError(
                "Config file misconfigured!\n\t"
                + "Either 'run_grid_search' or 'rt.run_train' should be True"
            )

    if rt.analyze:
        # TODO Add command line parameter to choose files
        if not rt.date_string:
            raise ValueError("Model date must be supplied for analysis!")

        execute_analysis(
            paths,
            config,
            rt,
            analyse_params,
            custom_objects,
            logger,
            rt.date_string,
            paths.results,
        )


if __name__ == "__main__":
    main()
