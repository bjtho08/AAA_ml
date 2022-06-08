"""Analysis executor
"""
import logging
import shutil
import sys
from pathlib import Path

import confuse
import numpy as np
import tensorflow as tf
import tifffile
from matplotlib import cm
from more_itertools import chunked
from PIL import Image
from skimage.io import imread
from tensorflow.keras.models import load_model
# local package
from tf_mmciad.utils.io import ImageAssembler, ImageTiler, float_to_palette
from tf_mmciad.utils.preprocessing import calculate_stats

from utils.utils import (CustomObjects, Parameters, get_filename, npa2s,
                           write_to_config)


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

    # model_path = next(Path(paths.models, model_date).glob("*val_jacc1*.h5"))
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

        # analysis_model = load_model(model_path, custom_objects=custom_objects.get())
        logger.info(f"{model_path = }")
        custom_objs = custom_objects.get()
        analysis_model = load_model(str(model_path), custom_objects=custom_objs)
        small_batch = 4
        try:
            for batch in chunked(slide_tiler.items(), small_batch):
                img_batch = np.array(
                    [(imread(fp) - pred_m) / pred_s for (fp, _) in batch]
                )
                pred_done = False
                prediction = analysis_model.predict(img_batch)
                pred_done = True
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
