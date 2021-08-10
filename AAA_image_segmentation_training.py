#!/usr/bin/env python
# coding: utf-8

# # AAA Image Segmentation
# This notebook will set up and either train a network or predict segmentations based on previous training

# ## Preliminary steps
# Load all the relevant libraries and set up some global parameters

import datetime
import logging
import os
import os.path as osp
import platform
import resource
import shutil
import sys
import traceback
from glob import glob

os.environ["TF_KERAS"] = "1"

import confuse
import matplotlib.pyplot as plt
import numpy as np


import talos as ta
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

import tqdm
import yaml

from keras_radam import RAdam
from skimage.io import imread, imsave
from sklearn.metrics import confusion_matrix
from talos.utils import early_stopper
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import ReLU
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import (
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    Nadam,
    RMSprop,
)
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tqdm import tqdm_notebook
from tqdm.keras import TqdmCallback

from keras_contrib.layers.advanced_activations.swish import Swish
from tf_mmciad.utils.callbacks import DeadReluDetector, PatchedModelCheckpoint
from tf_mmciad.utils.custom_loss import (
    categorical_focal_loss,
    get_weighted_categorical_crossentropy,
    jaccard1_coef,
    jaccard2_loss,
    tversky_loss,
    weighted_loss,
)
from tf_mmciad.utils.generator import DataGenerator, DataSet
from tf_mmciad.utils.hyper import prepare_for_talos

# local package
from tf_mmciad.utils.io import create_samples, load_slides_as_dict, move_files_in_dir
from tf_mmciad.utils.preprocessing import (
    augmentor,
    calculate_class_weights,
    calculate_stats,
    class_ratio,
)
from tf_mmciad.utils.u_net import u_net
from tf_mmciad.utils.u_resnet import u_resnet

print(f"Running Python version {platform.python_version()}")
print(f"Build {sys.version}")
print(f"TensorFlow version {tf.__version__}")

###################################
gpus = tf.config.experimental.list_physical_devices("GPU")
print("Num GPUs:", len(gpus))
# TensorFlow wizardry
# tf.debugging.set_log_device_placement(True)

# Don't pre-allocate memory; allocate as-needed

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Only allow a total of half the GPU memory to be allocated
# config.gpu_options.per_process_gpu_memory_fraction = 0.5

###################################


config = confuse.Configuration("AAAml", __name__)
config.set_file("config.yaml")

date_string = config["Runtime"]["date_string"].get()
# if date_string is None, current date will be used
clean_start = config["Runtime"]["clean_start"].get()
run_grid_search = config["Runtime"]["run_grid_search"].get()
run_train = config["Runtime"]["run_train"].get()
run_predict = config["Runtime"]["run_predict"].get()
run_eval = config["Runtime"]["run_eval"].get()
compute_class_weights = config["Runtime"]["compute_class_weights"].get()
debug = config["Runtime"]["debug"].get()
make_abundance_table = config["Runtime"]["make_abundance_table"].get()

class_map = config["class_map"].get()

class_colors = config["class_colors"].get()  # {

active_labels = [1, 2, 3, 4]
active_classes = [sorted(class_map, key=class_map.get)[i] for i in active_labels]
print(active_classes)
colorvec = np.asarray([class_colors[i] for i in active_labels])
active_colors = {class_: class_colors[class_] for class_ in active_labels}
num_cls = len(active_labels)
print(active_colors)
print(colorvec.shape)
ignore_cls = 0

dpaths = config["data"].get()
data_path = dpaths["path"]  # "data/"
wsi_path = dpaths["wsi"]  # "data/WSI/"
train_path = dpaths["train"]  # "training/"
val_path = dpaths["val"]  # "validation/"
test_path = dpaths["test"]  # "testing/"
model_storage = dpaths["models"]  # "models"
weight_path = dpaths["models"]  # "./weights/"
ignore_color = class_colors[0]
TILE_SIZE = (*config["input_meta"]["tile_size"].get(),)

if clean_start:
    # duplicate = [class_colors[label] for label in [2, 3, 6, 9, 11]]
    def get_color(name: str) -> list:
        return class_colors[class_map[name]]

    filter_dict = {
        "Ignore*": (get_color("Ignore*"), 0.9, 0.9, 1.0),
    }

    create_samples(
        osp.join(wsi_path), filter_dict, output_dir="../training", tile_size=TILE_SIZE
    )

patient_IDs = {
    osp.splitext(osp.basename(fname))[0].replace("train-", "")
    for fname in glob(osp.join(wsi_path, "train-*.tif"))
}
patient_patterns = [patient_ID + "*" for patient_ID in patient_IDs]


def file_counter(path: str) -> int:
    if osp.exists(path):
        return len(
            [name for name in os.listdir(path) if os.path.isfile(osp.join(path, name))]
        )
    return 0


ds = {
    "x": osp.join(data_path, train_path),
    "y": osp.join(data_path, train_path, "gt/"),
    "xv": osp.join(data_path, val_path),
    "yv": osp.join(data_path, val_path, "gt/"),
    "xt": osp.join(data_path, test_path),
    "yt": osp.join(data_path, test_path, "gt/"),
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
        file_counter(ds["x"]),
        ds["x"],
        file_counter(ds["y"]),
        ds["y"],
        file_counter(ds["xv"]),
        ds["xv"],
        file_counter(ds["yv"]),
        ds["yv"],
        file_counter(ds["xt"]),
        ds["xt"],
        file_counter(ds["yt"]),
        ds["yt"],
    )
)


if clean_start:
    train_m, train_s, x_min, x_max = calculate_stats(path="data/WSI/")  # , local=False)
    # TODO write to config
else:
    train_m = np.array(config["stats"]["train_m"].get())
    train_s = np.array(config["stats"]["train_s"].get())
    x_min = np.array(config["stats"]["x_min"].get())
    x_max = np.array(config["stats"]["x_max"].get())

print(f"Dataset\n-------\nMean:    {train_m:}\nSt.dev.: {train_s:}")
print(f"Min:     {x_min}\nMax:     {x_max}")

if debug:
    import imgaug as ia
    from imgaug import augmenters as iaa

    plt.rcParams["figure.figsize"] = (20.0, 16.0)
    mytest = imread("./data/train/gt/X_1001.tif")
    warp = iaa.PiecewiseAffine(scale=0.05, nb_rows=6, nb_cols=6, mode="reflect")
    trans = iaa.ElasticTransformation(alpha=80, sigma=(8.0), mode="reflect")
    warp_label = warp.augment_image(mytest)
    twarp_label = trans.augment_image(warp_label)
    plt.subplot(1, 3, 1)
    plt.imshow(mytest)
    plt.title("label")
    plt.subplot(1, 3, 2)
    plt.imshow(warp_label)
    plt.title("warp")
    plt.subplot(1, 3, 3)
    plt.imshow(twarp_label)
    plt.title("warp+elastic")


if compute_class_weights:
    # Calculate the class weights for the training data set
    # Optionally exclude a label by settings its weight to 0 using the ignore=label option
    cls_wgts = calculate_class_weights(
        data_path, active_labels, class_colors, ignore=ignore_cls
    )
    # TODO write to config
    print("cls_wgts:\n", cls_wgts)
    class_ratios = class_ratio(data_path, active_labels, class_colors)
    print("class_ratios:\n", class_ratios)
else:
    # Replace with a read from config later
    cls_wgts = {i: 1 if i != ignore_cls else 0 for i in active_labels}


if compute_class_weights:
    from functools import partial
    from tf_mmciad.utils.custom_loss import w_categorical_crossentropy

    cls_wgts = {
        0: 0.6462842253714038,
        1: 4.277279367175925,
        2: 0.7650382887989682,
        3: 2.3366233721836207,
        5: 0.38495865467512297,
        6: 11.749891543465838,
        7: 0.7357313031667244,
        9: 3.3815931126719256,
        10: 107.61779250337312,
        11: 0.5449941506459005,
        12: 0,
    }
    w_array = np.ones((num_cls, num_cls))
    for cls_id, weight in enumerate(cls_wgts.values()):
        w_array[cls_id, :] = weight  # populate False negatives
    w_array[:, -1] = 100.0  # Increase False Negative penalty for IGNORE class
    np.fill_diagonal(w_array, 1)  # Populate True positives
    w_array[-1, :] = 0.0  # Remove False positives penalty for IGNORE class
    # w_array_t = w_array.swapaxes(0, 1)
    w_array[2, 7] = 70  # Guessing Epithelium as Dysplasia
    w_array[2, 9] = 80  # Guessing Epithelium as Cancer
    w_array[9, 2] = 100  # Guessing Cancer as Epithelium
    w_array[7, 2] = 90  # Guessing Dysplasia as Epithelium
    w_array[4, 6] = 2  # Guessing Stroma as Inflammation
    w_array[4, 9] = 40  # Guessing Stroma as Cancer
    w_array[6, 4] = 4  # Guessing Inflammation as Stroma
    w_array[6, 9] = 30  # Guessing Inflammation as Cancer

    r"""
        Example weight matrix

     -> False positives
     v  False negatives
     \ True positives, always 1

                    True class
    P
    r    ___|  A  |  B  |  C  |  D  |  E
    e c   A |  1  | 3.0 | 0.5 | 1.7 | 0.3 
    d l   B | 0.6 |  1  | 0.5 | 1.7 | 0.3
    i a   C | 0.6 | 3.0 |  1  | 1.7 | 0.3
    c s   D | 0.6 | 3.0 | 0.5 |  1  | 0.3
    t s   E | 0.6 | 3.0 | 0.5 | 1.7 |  1
    e
    d
    """
    with np.printoptions(precision=3, suppress=True, linewidth=100):
        print(w_array)
    w_ce = partial(w_categorical_crossentropy, weights=w_array)
    w_ce.__name__ = "weighted_categorical_crossentropy"


if compute_class_weights:
    w_TL = weighted_loss(tversky_loss, cls_wgts)
    # w_cat_CE = weighted_loss(categorical_crossentropy, cls_wgts)
    # w_cat_CE = get_weighted_categorical_crossentropy(weights=[v for v in cls_wgts.values()])
    w_TL.__name__ = "w_TL"
    # w_cat_CE.__name__ = "w_cat_CE"


# I/O Params

IMG_ROWS, IMG_COLS, IMG_CHANNELS = (None, None, config["input_meta"]["channels"].get())
# architecture params
NB_FILTERS_0 = config["nb_filters_0"].get()
SIGMA_NOISE = 0.01

# ****  deep learning model
SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
BATCH_SIZE = config["batch_size"].get()
NB_EPOCH = config["nb_epoch"].get()
NB_FROZEN = config["nb_frozen"].get()
VERBOSE = config["verbose"].get()

# ****  train
if not date_string:
    date_string = str(datetime.date.today())

if run_grid_search:
    def predefined_generator(path: str, augment: bool = True):
        return DataGenerator(
            path,
            active_colors,
            train_m,
            train_s,
            x_min,
            x_max,
            batch_size=BATCH_SIZE,
            dim=TILE_SIZE,
            n_channels=config["input_meta"]["channels"].get(),
            n_classes=config["statics"]["num_cls"].get(),
            shuffle=True,
            augment=augment,
        )


    train_generator = predefined_generator(train_path)

    val_generator = predefined_generator(val_path, augment=False,)

    #     statics = {
    #         "shape": SHAPE,
    #         "nb_epoch": 50,#NB_EPOCH,
    #         #"nb_frozen": NB_FROZEN,
    #         "nb_filters_0": NB_FILTERS_0,
    #         "batch_size": BATCH_SIZE,
    #         "verbose": VERBOSE,
    #         "num_cls": num_cls,
    #         "batchnorm": True,
    #         "maxpool": False,
    #         "date": date_string,
    #         #"opt": Adam,
    #         "depth": 4,
    #         #"arch": "U-Net",
    #         "dropout": 0,
    #         "decay": 0.0,
    #         "sigma_noise": 0,
    #         #"act": 'relu',
    #         "pretrain": 0,
    #         "lr": 1e-4,
    #         "class_weights": False,
    #         #"loss_func": "cat_CE",
    #         #"init": "he_normal",
    #     }
    statics = config["statics"].get()
    statics["date"] = date_string
    p = config["grid_params"].get()
    # fit params
    #     p = {
    #         #"dropout": [0],
    #         #"decay": [0.0],
    #         #"lr": [1e-3, 1e-4, 1e-5],
    #         #"sigma_noise": [0],
    #         "nb_filters_0": [12, 16, 32, 64],
    #         #"pretrain": [0, 2, 4],
    #         #"class_weights": [True, False],
    #         "loss_func": ["cat_CE", "tversky_loss", "cat_FL"],
    #         "arch": ["U-Net"],
    #         "act": ["swish", "relu"],
    #         "opt": ["adam",],
    #         "init": ["he_normal",]# "glorot_uniform"]
    #     }

    talos_model = prepare_for_talos(
        model_storage, cls_wgts, statics, train_generator, val_generator, debug=debug
    )

    dummy_x = np.empty((1, BATCH_SIZE, 384, 384))
    dummy_y = np.empty((1, BATCH_SIZE))

    scan_object = ta.Scan(
        x=dummy_x,
        y=dummy_y,
        disable_progress_bar=False,
        print_params=True,
        model=talos_model,
        params=p,
        experiment_name=dpaths["grid"] + date_string,
        # reduction_method='gamify',
        allow_resume=True,
    )

analyze_object = ta.Analyze(scan_object)

test_slides, test_targets = load_slides_as_dict(
    "data/WSI/", "*3295 5V", train_m, train_s, 255, "gt", num_cls, colorvec
)

if run_predict:
    statics = config["statics"].get()
    statics["date"] = date_string
    
