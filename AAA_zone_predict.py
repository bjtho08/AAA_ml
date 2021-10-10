"""AAA Image Segmentation
This script will set up and either train a network or predict segmentations based on
previous training
"""
import fileinput
import logging
import platform
import re
import shutil
import sys
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Optional, Union

import confuse
import numpy as np
import silence_tensorflow
import tensorflow as tf
import tifffile
from matplotlib import cm
from more_itertools import chunked
from PIL import Image
from skimage.io import imread
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.optimizers import Adam

# local package
from tf_mmciad.model.layers import MyConv2D
from tf_mmciad.utils.f_scores import F1Score
from tf_mmciad.utils.custom_loss import (
    categorical_focal_loss,
    jaccard1_coef,
    jaccard2_coef,
    jaccard1_loss,
    jaccard2_loss,
    tversky_loss,
    weighted_loss,
)
from tf_mmciad.utils.generator import DataGenerator
from tf_mmciad.utils.hyper import TalosModel, single_run
from tf_mmciad.utils.io import (
    ImageAssembler,
    ImageTiler,
    create_samples,
    float_to_palette,
)
from tf_mmciad.utils.preprocessing import (
    calculate_class_weights,
    calculate_stats,
    class_ratio,
)
from tf_mmciad.utils.swish import Swish

import talos as ta


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
            print(re.sub(r"(?<=" + flag + r": )(true|false)", str(state).lower(), line.rstrip()))

def write_to_config(subs: Optional[Dict[str, float]]=None):
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
                        r"\g<1>" + np.array2string(val, separator=", ").replace("\n", ""),
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


def execute_grid_search(config, rt, paths, params, tb_callback_params, logger: logging.Logger):
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


def execute_single_train(config, rt, paths, params, tb_callback_params, logger: logging.Logger, error_resume: bool):
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


def execute_predict(
    paths: confuse.AttrDict,
    rt: confuse.AttrDict,
    patient_val_ids: List[str],
    params: Parameters,
    custom_objects: CustomObjects,
    train_date: str,
    logger: logging.Logger,
    scan_object: ta.Scan = None,
):
    """Runs the predict and test_val_ids on the RISA dataset .

    Args:
        paths ([type]): [description]
        rt ([type]): [description]
        patient_val_ids ([type]): [description]
        params (Parameters): [description]
        custom_objects (CustomObjects): [description]
        train_date (str): [description]
        logger (logging.Logger): [description]
        scan_object (ta.Scan, optional): [description]. Defaults to None.

    Raises:
        err: [description]
    """
    # Execute the final prediction on the train_date dataset .

    # Args:
    #    train_date (str): [description]
    #    from_talos (bool, optional): [description]. Defaults to True.

    # Raises:
    #    err: [description]
    logger.info("Running 'execute_predict' branch ...")
    tmp_path = Path(paths.data, "tmp")
    input_file = sorted(Path(paths.wsi).glob("*.tif"))
    target_file = sorted(Path(paths.wsi, "gt").glob("*.png"))

    input_file = [file for file in input_file if file.stem in patient_val_ids]
    target_file = [file for file in target_file if file.stem in patient_val_ids]
    logger.info(f"{target_file = }")

    if scan_object:
        sorted_models = scan_object.data.sort_values("jaccard1_coef", ascending=False)
        ranks = [sorted_models.iloc[i].name for i in range(len(sorted_models.index))]
        models = [Path(train_date + "_" + str(i) + ".h5") for i in ranks]
    else:
        models = [
            f for f in Path(paths.models, rt.date_string).iterdir()
            if f.is_dir() and 'model' in str(f)
            ]
        ranks = [ts.stem[6:] for ts in models]  # strip filepath of prefix and suffix

    for modelpath, rank in zip(models, ranks):
        for input_path, target_path in zip(input_file, target_file):
            input_name, target_name = (
                get_filename(input_path),
                get_filename(target_path),
            )
            tmp_slide_path = tmp_path / "results" / f"{train_date}_{rank}.png"
            fp = Path(paths.results, f"{train_date}_{rank}", f"{input_name}.png")
            if fp.is_file() and fp.stat().st_size > 0:
                logger.info(f"Skipping {fp.name}, already exists!")
                continue
            slide_tiler = ImageTiler(
                input_path, params.dim, tmp_path / input_name, force_extension="tif",
            )
            test_target = ImageTiler(
                target_path, params.dim, tmp_path / f"{target_name}_gt"
            )
            test_paths = (input_path, target_path)
            logger.info(f"{test_paths = }\n\t{input_name = }")
            fp.parent.mkdir(parents=True, exist_ok=True)

            if scan_object:
                test_model = model_from_json(
                    scan_object.saved_models[rank], custom_objects=custom_objects.get()
                )
                test_model.set_weights(scan_object.saved_weights[rank])
                test_model.compile(
                    Adam(), categorical_crossentropy, metrics=["acc", jaccard1_coef]
                )
            else:
                test_model = load_model(str(modelpath), custom_objects=custom_objects.get())
            small_batch = 4
            try:
                for batch in chunked(slide_tiler.items(), small_batch):
                    img_batch = np.array([imread(fp) for (fp, _) in batch])
                    pred_done = False
                    prediction = test_model.predict(img_batch)
                    pred_done = True
                    for (save_path, _), pred_tile in zip(batch, prediction):
                        tifffile.imwrite(
                            save_path,
                            pred_tile,
                            metadata={"axes": "XYC"},
                        )
            except ValueError:
                logger.info("Prediction already complete!")
                pred_done = True
            except tf.errors.ResourceExhaustedError as err:
                logger.exception("OOM Error during prediction")
                logger.debug("Debugging information:")
                logger.debug(f"batch = {[p[0] for p in batch]}")
                metadata = {
                    method: getattr(img_batch, method)
                    for method in dir(img_batch)
                    if (not callable(getattr(img_batch, method)))
                    and (not method.startswith("__"))
                }
                for attr, value in metadata.items():
                    logger.debug(f"img_batch.{attr} = {value}")
                raise err from None
            if not fp.is_file() and pred_done:
                # logger.info("Writing to tmp directory ...")
                img_assembler = ImageAssembler(tmp_slide_path, slide_tiler)
                img_assembler.merge(colors=params.colorvec, format="PNG", mode="P")
                del img_assembler
                # logger.info("Finished writing to tmp directory")

            # copy target and prediction images to result folder
            target_fp = fp.with_name(fp.stem + "_00target.png")
            pred_fp = fp.with_name(fp.stem + "_01prediction.png")
            logger.info(f"Copying {target_path} to {target_fp}")
            shutil.copyfile(target_path, target_fp)
            logger.info(f"Moving {tmp_slide_path} to {pred_fp}")
            shutil.move(tmp_slide_path, pred_fp)

            # TODO add functionality to ImageAssembler.merge()
            if rt.create_probability_maps:
                # Create multi-channel TIF of probability maps
                # logger.info("Writing to disk ...")
                prob_path = fp.parent
                prob_path = prob_path / "probs" / fp.name
                prob_path.parent.mkdir(exist_ok=True)
                prob_assembler = ImageAssembler(
                    prob_path.with_suffix(".tif"), slide_tiler
                )
                prob_assembler.merge_multichannel()
                del prob_assembler
                # logger.info("Writing to disk ... Finished")

                # Create custom palettes for indexed png images
                plasma_map = cm.get_cmap("plasma")
                plasma_pal = (
                    np.array(
                        (plasma_map(np.arange(256) / 255.0)[..., :3] * 255) // 1,
                        dtype=np.uint8,
                    )
                    .flatten()
                    .tolist()
                )

                # load prediction prob channels, convert to PIL image and
                # add custom palette
                for cls_, cls_name in enumerate(params.class_names):
                    # extract probability maps class-wise
                    prob_channel = prob_path.with_name(
                        prob_path.stem + f"_ch_{cls_:0>2}" + prob_path.suffix
                    )
                    img = tifffile.imread(prob_channel)
                    pil_img = Image.fromarray(img)
                    # Convert maps to uint8
                    pil_img = float_to_palette(pil_img)
                    # Create PIL image from maps and add color palette
                    pil_img.putpalette(plasma_pal)
                    # Save to results folder
                    pil_img.save(
                        fp.with_name(fp.stem + f"{(cls_ + 2):02d}{cls_name}.png"),
                        compress_level=1,
                    )
            slide_tiler.remove()
            test_target.remove()
            del test_model, slide_tiler, test_target


def execute_analysis(
    paths: confuse.AttrDict,
    config: confuse.Configuration,
    rt: confuse.AttrDict,
    params: Parameters,
    custom_objects: CustomObjects,
    logger: logging.Logger,
    model_date: str,
    output_dir: str,
):
    """Execute the analysis of the given dataset using the given model.

    Args:
        paths ([type]): [description]
        config ([type]): [description]
        rt ([type]): [description]
        params (Parameters): [description]
        logger (logging.Logger): [description]
        model_date (str): [description]
        output_dir (str): [description]

    Raises:
        err: [description]
    """
    logger.info("Running 'execute_analysis' branch ...")
    if rt.compute_stats:
        pred_m, pred_s, *_ = calculate_stats(path=paths.analysis, prefix="")
        pred_stats = {"pred_m": pred_m, "pred_s": pred_s}
        write_to_config(pred_stats)
    else:
        pred_m = np.array(config["stats"]["pred_m"])
        pred_s = np.array(config["stats"]["pred_s"])
        # pred_m = np.array(config["stats"]["train_m"])
        # pred_s = np.array(config["stats"]["train_s"])

    print(
        f"Dataset\n-------\nMean:    {npa2s(pred_m, separator=', '):}\n",
        f"St.dev.: {npa2s(pred_s, separator=', '):}",
    )

    params.mean, params.stdev = pred_m, pred_s
    tmp_path = Path(paths.data, "tmp")
    input_files = sorted(Path(paths.analysis).glob("*.tif"))

    #model_path = next(Path(paths.models, model_date).glob("*val_jacc1*.h5"))
    model_path = Path(paths.models, model_date, "saved_model")

    for input_path in input_files:
        input_name = get_filename(input_path)
        logger.debug(f"{input_path = },\n\t{input_name = }")
        output_path = Path(output_dir, model_date, input_name + "_prediction.png")
        if output_path.is_file() and output_path.stat().st_size > 0:
            logger.info(f"Skipping {output_path.name}, already exists!")
            continue
        slide_tiler = ImageTiler(
            input_path, params.dim, tmp_path / input_name, force_extension="tif",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_slide_path = tmp_path / "results" / f"{model_date}.png"
        sys.stderr.flush()

        #analysis_model = load_model(model_path, custom_objects=custom_objects.get())
        logger.info(f"{model_path = }")
        custom_objs = custom_objects.get()
        analysis_model = load_model(str(model_path), custom_objects=custom_objs)
        small_batch = 4
        try:
            for batch in chunked(slide_tiler.items(), small_batch):
                img_batch = np.array([(imread(fp) - pred_m) / pred_s for (fp, _) in batch])
                pred_done = False
                prediction = analysis_model.predict(img_batch)
                pred_done = True
                mini_tile = prediction[0, :256, :256]
                #continue
                for (save_path, _), pred_tile in zip(batch, prediction):
                    tifffile.imwrite(
                        save_path,
                        np.moveaxis(pred_tile, -1, 0),
                        photometric="minisblack",
                        metadata={"axes": "CXY"},
                    )
                    # raise KeyboardInterrupt("Debug")
        except ValueError:
            logger.info("Prediction already complete!")
            pred_done = True
        except tf.errors.ResourceExhaustedError as err:
            logger.exception("OOM Error during prediction")
            logger.debug("Debugging information:")
            logger.debug(f"batch = {[p[0] for p in batch]}")
            metadata = {
                method: getattr(img_batch, method)
                for method in dir(img_batch)
                if (not callable(getattr(img_batch, method)))
                and (not method.startswith("__"))
            }
            for key, val in metadata.items():
                logger.debug(f"img_batch.{key} = {val}")
            raise err from None
        # continue
        if not output_path.is_file() and pred_done:
            # logger.info("Writing to tmp directory ...")
            img_assembler = ImageAssembler(tmp_slide_path, slide_tiler)
            img_assembler.merge(colors=params.colorvec, format="PNG", mode="P")
            del img_assembler
            # logger.info("Finished writing to tmp directory")

            # copy prediction images to result folder
            logger.info(f"Moving {tmp_slide_path} to {output_path}")
            shutil.move(tmp_slide_path, output_path)
            sys.stderr.flush()

        # TODO add functionality to ImageAssembler.merge()
        if rt.create_probability_maps:
            # Create multi-channel TIF of probability maps
            # logger.info("Writing to disk ...")
            prob_path = output_path.parent
            prob_path = prob_path / "probs" / output_path.name
            prob_path.mkdir(exist_ok=True)
            prob_assembler = ImageAssembler(prob_path.with_suffix(".tif"), slide_tiler)
            prob_assembler.merge_multichannel()
            del prob_assembler
            # logger.info("Writing to disk ... Finished")
            sys.stderr.flush()

            # Create custom palettes for indexed png images
            plasma_map = cm.get_cmap("plasma")
            plasma_pal = (
                np.array(
                    (plasma_map(np.arange(256) / 255.0)[..., :3] * 255) // 1,
                    dtype=np.uint8,
                )
                .flatten()
                .tolist()
            )

            # load prediction prob channels, convert to PIL image and
            # add custom palette
            for cls_, cls_name in enumerate(params.class_names):
                # extract probability maps class-wise
                prob_channel = prob_path.with_name(
                    prob_path.stem + f"_ch_{cls_:0>2}" + prob_path.suffix
                )
                img = tifffile.imread(prob_channel)
                pil_img = Image.fromarray(img)
                # Convert maps to uint8
                pil_img = float_to_palette(pil_img)
                # Create PIL image from maps and add color palette
                pil_img.putpalette(plasma_pal)
                # Save to results folder
                pil_img.save(
                    output_path.with_name(
                        output_path.stem + f"{(cls_ + 2):02d}{cls_name}.png"
                    ),
                    compress_level=1,
                )
            slide_tiler.remove()
            del analysis_model, slide_tiler


def main():
    """Runs the main program and prints the output
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
        'appName': str,
        'Runtime': {
            'clean_start': bool,
            'compute_stats': bool,
            'compute_class_weights': bool,
            'date_string': str,
            'debug': bool,
            'make_abundance_table': bool,
            'run_grid_search': bool,
            'run_train': bool,
            'run_predict': bool,
            'run_eval': bool,
            'analyze': bool,
            'create_probability_maps': bool,
        },
        'data': confuse.MappingValues(confuse.OneOf([str, confuse.Sequence([str])])),
        'class_map': confuse.MappingValues(int),
        'class_colors': confuse.MappingValues(confuse.Sequence([int])),
        'active_labels': confuse.Sequence([int]),
        'ignore_cls': confuse.Optional(int, default=None),
        'stats': confuse.MappingValues(confuse.Sequence([float])),
        'input_meta': {
            'x': int,
            'y': int,
            'channels': int,
            'tile_size': confuse.Sequence([int]),
            'shape': confuse.Sequence([confuse.Optional(int, default=None)]),
        },
        'batch_size': int,
        'nb_epoch': int,
        'nb_frozen': int,
        'verbose': int,
        'drop': int,
        'statics': {
            'shape': confuse.Sequence([confuse.Optional(int, default=None)]),
            'nb_epoch': int,
            'nb_filters_0': int,
            'batch_size': int,
            'verbose': int,
            'num_cls': int,
            'batchnorm': bool,
            'maxpool': bool,
            'opt': str,
            'depth': int,
            'arch': str,
            'dropout': float,
            'decay': float,
            'sigma_noise': float,
            'act': str,
            'pretrain': int,
            'lr': float,
            'class_weights': bool,
            'loss_func': str,
            'init': str,
        },
        'grid_params': {
            'nb_filters_0': confuse.Sequence([int]),
            'depth': confuse.Sequence([int]),
            'loss_func': confuse.Sequence([str]),
            'arch': confuse.Sequence([str]),
            'act': confuse.Sequence([str]),
            'opt': confuse.Sequence([str]),
            'init': confuse.Sequence([str]),
        }
    }

    config = confuse.Configuration("AAAml", __name__)
    config.set_file("config.yaml")
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
        def get_color(name: str) -> list:
            """Returns as a list the RGB color value corresponding to the given name.

            Args:
                name (str): [description]

            Returns:
                list: [description]
            """
            return class_colors[class_map[name]]

        filter_dict = {
            "None": (class_map["None"], 0.8, 0.9, None),
            "Ignore*": (class_map["Ignore*"], 0.8, 0.9, None),
        }
        for p in Path(paths.train).glob("**/*." + target_ext):
            p.unlink()
        create_samples(
            Path(paths.wsi),
            filter_dict,
            prefix="",
            output_dir="../training",
            tile_size=tile_size,
            #duplication_list=class_map["Thrombus"],
        )
        rewrite_config_flag("clean_start", False)

    # Denne liste skal indeholde alle tiles der matcher ID'et fra patient_val_ids
    # bsf burde blive defineret dynamisk.
    # En tilsvarende liste skal genereres for val_ids
    tiles = sorted(Path(paths.train).glob("*." + target_ext))
    train_ids = [
        tile.stem for tile in tiles
        if not re.match(r'('+'|'.join(patient_val_ids)+r')_[0-9]+', tile.stem)
    ]
    val_ids = [
        tile.stem for tile in tiles
        if re.match(r'('+'|'.join(patient_val_ids)+r')_[0-9]+', tile.stem)
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
        # w_cat_CE = weighted_loss(categorical_crossentropy, cls_wgts)
        # w_cat_CE = get_weighted_categorical_crossentropy(
        #   weights=[v for v in cls_wgts.values()]
        # )
        w_tversky_loss.__name__ = "w_TL"
        # w_cat_CE.__name__ = "w_cat_CE"

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
        val_ids=patient_val_ids,
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
            f for f in Path(paths.models, rt.date_string).iterdir()
            if f.is_dir() and 'model' in str(f)
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
        # from mmciad.utils.preprocessing import merge_labels
        # TODO Add command line parameter to choose files
        if not rt.date_string:
            rt.date_string = str(date.today())

        if not rt.run_grid_search and rt.run_train:
            execute_predict(
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
                execute_predict(
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

    if rt.analyze:
        # from mmciad.utils.preprocessing import merge_labels
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
