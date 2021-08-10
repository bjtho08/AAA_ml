import os
import numpy as np

from tensorflow.python.ops.gen_math_ops import ceil

os.environ["TF_KERAS"] = "1"

import sys
import shutil
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

from tf_mmciad.utils.swish import Swish
from tf_mmciad.utils.u_net import u_net
from tf_mmciad.utils.callbacks import PatchedModelCheckpoint, TensorBoardWrapper
from tf_mmciad.utils.generator import DataGenerator

###################################
gpus = tf.config.experimental.list_physical_devices("GPU")
print("Num GPUs:", len(gpus), file=sys.stderr, flush=True)

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
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

###################################

IMG_SIZE= 384
N_CHANNELS = 3
AUTOTUNE = tf.data.experimental.AUTOTUNE

def resizer(image, label):
    resize = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
        layers.experimental.preprocessing.Rescaling(1.0/255.0)
    ])

    return resize(image), label

def categorizer(labels, n_classes: int=1000):
    print(labels)
    return keras.utils.to_categorical(labels, num_classes=n_classes)

# (train_ds, val_ds), ds_info = tfds.load(
#     'mnist',
#     split=['train', 'test'],
#     batch_size=64,
#     as_supervised=True,
#     shuffle_files=True,
#     with_info=True,
# )
# train_ds = train_ds.map(resizer,
#     num_parallel_calls=AUTOTUNE)
# val_ds = val_ds.map(resizer,
#     num_parallel_calls=AUTOTUNE)

# print(dir(val_ds))
# class ds_gen(keras.utils.Sequence):
#     def __init__(self, ds, b_size) -> None:
#         self.ds = ds
#         self.ds_np = tfds.as_numpy(ds)
#         self.it = iter(self.ds_np)
#         self.batch_size = b_size
#         self.data = [ex for ex in self.it]
#         self._last_index = 0

#     def __len__(self):
#         return len(list(self.ds.as_numpy_iterator()))//self.batch_size

#     def __getitem__(self, index):
#         self._last_index = index
#         return self.data[index]
    
#     def __next__(self):
#         return self.__getitem__(self._last_index+1)

#     def on_epoch_begin(self):
#         self._last_index = 0

# test_val_gen = ds_gen(val_ds, 64)
# print(len(test_val_gen))
# print(next(test_val_gen)[1])
# for i in range(len(test_val_gen)-1):
#     next(test_val_gen)
# test_val_gen.on_epoch_begin()
# print(next(test_val_gen)[1])

class_colors = {
  0: [  0,   0,   0],
  1: [180, 180, 180],
  2: [  0,   0, 255],
  3: [  0, 255,   0],
  4: [255,   0,   0],
}
active_labels = [1, 2, 3, 4]
active_colors = {class_: class_colors[class_] for class_ in active_labels}
means = np.array([210.52194532, 196.59868844, 201.91357681]) # Data set mean
stds = np.array([32.27907747, 52.38655931, 41.60488245]) # Data set standard deviation
x_min = np.array([1., 1., 1.]) # Data set minimum value - always zero for 8-bit
x_max = np.array([255., 247., 254.]) # Data set maximum value - always 255 for 8-bit
path = "data/testing/"
color_dict = active_colors
tb_callback_params = {
    "means": means,
    "stds": stds,
    "x_min": x_min,
    "x_max": x_max,
    "path": path,
    "color_dict": color_dict,
    "batch_size": 7,
    "dim": (384, 384),
    "n_channels": 3,
    "n_classes": 4,
}

gen_params = {
    "batch_size": 12,
    "dim": (384, 384),
    "n_channels": 3,
    "n_classes": 4,
}

train_args = [
    "data/training/", color_dict, means, stds, x_min, x_max
]
tb_args = [
    path, color_dict, means, stds, x_min, x_max
]
train_gen = DataGenerator(
    *train_args,
    **gen_params,
    shuffle=False,
    augment=False,
)

class KDebug(Callback):
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def eprint(*msg):
        print(*msg, file=sys.stderr)
    
    def on_batch_begin(self, batch, logs):
        self.eprint(dir(model))
        self.eprint(self.model.input_shape, self.model.output_shape)
        self.eprint(batch)
        

class LogSegmentationProgress(Callback):
    """Simple image writer class"""

    def __init__(self, file_writer_cm, tensorboard_params: dict):
        super().__init__()
        self.file_writer = file_writer_cm
        self.tb_params = tensorboard_params
        self.path = self.tb_params.pop("path")
        self.color_dict = self.tb_params.pop("color_dict")
        self.means = self.tb_params.pop("means")
        self.stds = self.tb_params.pop("stds")
        self.x_min = self.tb_params.pop("x_min")
        self.x_max = self.tb_params.pop("x_max")
        self.tb_args = [
            self.path, self.color_dict, self.means, self.stds, self.x_min, self.x_max
        ]

    def on_epoch_end(self, epoch, logs=None):
        _ = logs
        
        test_generator = DataGenerator(
            *self.tb_args,
            **self.tb_params,
            shuffle=False,
            augment=False,
        )
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.model.predict(test_generator)
        test_pred = np.argmax(test_pred_raw, axis=-1)
        # Read the input and target images
        norm_input, targets = test_generator[0]
        # revert normalized input to raw RBG
        raw_input = (norm_input * self.stds) + self.means
        raw_input = np.round(raw_input).astype(np.uint8)
        # recreate color matrix
        palette = np.array(
            list(self.color_dict.values()),
            dtype="uint8",
        )
        # convert one-hot encoded matrices to RBG
        cat_pred = palette[test_pred]
        cat_targets = palette[np.argmax(targets, axis=-1)]
        # Log the image summaries.
        with self.file_writer.as_default():
            tf.summary.image("Raw input", raw_input, max_outputs=8, step=epoch)
            tf.summary.image("Ground truth", cat_targets, max_outputs=8, step=epoch)
            tf.summary.image("Prediction", cat_pred, max_outputs=8, step=epoch)


model = u_net(
    (IMG_SIZE, IMG_SIZE, N_CHANNELS),
    nb_filters=32,
    conv_size=3,
    initialization="he_normal",
    activation=Swish,
    output_channels=4,
    batchnorm=True,
    maxpool=False,
    sigma_noise=0,
    arch="U-Net",
    encode_only=False,
)

model.compile(
    keras.optimizers.Adam(1.0e-4),
    loss=keras.losses.categorical_crossentropy,
    metrics=['acc', ],
)

shutil.rmtree("logs/aaa_toynet/")
print(f"""{os.path.exists("logs/aaa_toynet/")=}""", file=sys.stderr)
time.sleep(10)
os.makedirs("logs/aaa_toynet/", exist_ok=True)
csv_path = "logs/aaa_toynet/metrics.csv"
csv_logger = CSVLogger(csv_path, append=True)

tb_path = "logs/aaa_toynet/tb"
# val_gen_iter = ds_gen(val_ds, 64)
# tb_callback = TensorBoardWrapper(
#     val_gen_iter,
#     log_dir=tb_path,
#     histogram_freq=1,
#     embeddings_freq=0,
#     update_freq="epoch",
# )
tb_callback = TensorBoard(
    log_dir=tb_path,
    histogram_freq=1,
    embeddings_freq=0,
    update_freq="epoch",
)

early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=0.0001, patience=15, verbose=0, mode="auto"
)
reduce_lr_on_plateau = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=5, min_lr=1e-8, verbose=1
)
pbar = tfa.callbacks.TQDMProgressBar()
os.makedirs("models/aaa_toynet/", exist_ok=True)
model_checkpoint = PatchedModelCheckpoint(
    "models/aaa_toynet/pretrain_epoch_{epoch}_acc_{acc:0.4f}.h5",
    verbose=0,
    monitor="acc",
    save_best_only=True,
)

file_writer_seg = tf.summary.create_file_writer(tb_path + "/images")

log_image_segmentation = LogSegmentationProgress(
    file_writer_seg, tb_callback_params
)

history = model.fit(
    x=train_gen,
    epochs=6,
    steps_per_epoch=8,
    workers=4,
    verbose=2,
    callbacks=[
        pbar,
        model_checkpoint,
        tb_callback,
        log_image_segmentation,
    ],
)
print(history.history)
