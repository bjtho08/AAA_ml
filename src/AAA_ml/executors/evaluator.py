"""Evaluation executor
"""
import logging
from pathlib import Path
from typing import Callable

import confuse
import tensorflow as tf
from tensorflow.keras.models import load_model
# local package
from tf_mmciad.utils.custom_loss import jaccard1_coef

from utils.utils import CustomObjects, Parameters, predefined_datagenerator


def execute_evaluation(
    paths: confuse.AttrDict,
    params: Parameters,
    custom_objects: CustomObjects,
    train_date: str,
    logger: logging.Logger,
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
    def config_jac_coef(min_class: int = 1) -> Callable:
        def jaccard1_eval_coef(y_true: tf.Tensor, y_pred: tf.Tensor):
            """Jaccard index coefficient

            :param y_true: true label
            :type y_true: int
            :param y_pred: predicted label
            :type y_pred: int or float
            :param smooth: smoothing parameter, defaults to SMOOTH
            :type smooth: float, optional
            :return: Jaccard coefficient
            :rtype: float
            """
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            weights = tf.cast(tf.greater_equal(y_true_f, min_class), tf.float32)
            y_true_f = y_true_f * weights
            y_pred_f = y_pred_f * weights
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            union = (
                tf.keras.backend.sum(y_true_f)
                + tf.keras.backend.sum(y_pred_f)
                - intersection
            )
            return (intersection + 1.0) / (union + 1.0)

        return jaccard1_eval_coef

    logger.info("Running 'execute_evaluation' branch ...")
    # train_generator = predefined_datagenerator(
    #     paths.train, params=params, id_list=params.train_ids
    # )
    train_generator = predefined_datagenerator(
        paths.train, params=params, id_list=params.train_ids
    )
    val_generator = predefined_datagenerator(
        paths.train, params=params, augment=False, id_list=params.val_ids
    )

    model_path = Path(paths.models, train_date, "saved_model")
    jac_from_1 = config_jac_coef(1)
    jac_from_1.__name__ = "jac_from_1"
    jac_from_2 = config_jac_coef(2)
    jac_from_2.__name__ = "jac_from_2"
    eval_model: tf.keras.Model = load_model(
        str(model_path), custom_objects=custom_objects.get()
    )
    eval_model.compile(metrics=["acc", jaccard1_coef, jac_from_1, jac_from_2])

    eval_dict = {}
    eval_dict["val"] = eval_model.evaluate(val_generator)
    eval_dict["train"] = eval_model.evaluate(train_generator)

    return eval_dict
