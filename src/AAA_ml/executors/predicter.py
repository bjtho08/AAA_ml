"""Prediction executor
"""
import logging
import shutil
from pathlib import Path
from typing import List

import confuse
import numpy as np

import tensorflow as tf
import tifffile
from matplotlib import cm
from more_itertools import chunked
from PIL import Image
from skimage.io import imread
from sklearn.metrics import (classification_report, confusion_matrix,
                             jaccard_score)
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.optimizers import Adam
# local package
import talos as ta
from tf_mmciad.utils.custom_loss import jaccard1_coef
from tf_mmciad.utils.io import ImageAssembler, ImageTiler, float_to_palette

from utils.utils import CustomObjects, Parameters, get_filename, write_results


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
    all_pred_dict = {}

    if scan_object:
        sorted_models = scan_object.data.sort_values("jaccard1_coef", ascending=False)
        ranks = [sorted_models.iloc[i].name for i in range(len(sorted_models.index))]
        models = [Path(train_date + "_" + str(i) + ".h5") for i in ranks]
    else:
        models = [
            f
            for f in Path(paths.models, rt.date_string).iterdir()
            if f.is_dir() and "model" in str(f)
        ]
        ranks = [
            ts.stem if ts.is_file() else ts.name for ts in models
        ]  # strip filepath of prefix and suffix

    for modelpath, rank in zip(models, ranks):
        y_true = []
        y_pred = []
        logger.info(f"{modelpath = }")
        for input_path, target_path in zip(input_file, target_file):
            input_name, target_name = (
                get_filename(input_path),
                get_filename(target_path),
            )
            tmp_slide_path = tmp_path / "results" / f"{train_date}_{rank}.png"
            fp = Path(paths.results, train_date, rank, f"{input_name}.png")
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
                test_model = load_model(
                    str(modelpath), custom_objects=custom_objects.get()
                )
            small_batch = 4
            try:
                for batch in chunked(slide_tiler.items(), small_batch):
                    img_batch = np.array(
                        [(imread(tp) - params.mean) / params.stdev for (tp, _) in batch]
                    )
                    pred_done = False
                    prediction = test_model.predict(img_batch)
                    pred_done = True
                    for (save_path, _), pred_tile in zip(batch, prediction):
                        tifffile.imwrite(
                            save_path,
                            np.moveaxis(pred_tile, -1, 0),
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
            y_true.append(np.array(Image.open(target_fp)).flatten())
            y_pred.append(np.array(Image.open(pred_fp)).flatten())

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
        y_true = np.concatenate(y_true).astype(np.uint8).flatten()
        y_pred = np.concatenate(y_pred).astype(np.uint8).flatten()
        jacc_score_ignore = jaccard_score(
            y_true, y_pred, labels=[1, 2, 3, 4, 5], average="weighted"
        )
        jacc_score_noignore = jaccard_score(
            y_true, y_pred, labels=[2, 3, 4, 5], average="weighted"
        )
        report = classification_report(
            y_true,
            y_pred,
            labels=[1, 2, 3, 4, 5],
            target_names=["Ignore", "Zone 1", "Zone 2", "Thrombus", "Background"],
            output_dict=True,
        )
        conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
        pred_dict = {}
        pred_dict["Weighted Jaccard score with Ignore class"] = jacc_score_ignore
        pred_dict["Weighted Jaccard score without Ignore class"] = jacc_score_noignore
        pred_dict["Classification report"] = report
        pred_dict["Confusion matrix"] = conf_matrix
        save_path = Path(paths.results, rt.date_string, "results.txt")
        if not save_path.exists():
            write_results(save_path, rank, pred_dict)
        else:
            write_results(save_path, rank, pred_dict, mode='a')
        all_pred_dict[rank] = pred_dict

    return all_pred_dict
