import os

os.environ["TF_KERAS"] = "1"

import sys
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

from tf_mmciad.utils.swish import Swish
from tf_mmciad.utils.u_net import u_net
from tf_mmciad.utils.callbacks import PatchedModelCheckpoint

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

IMG_SIZE= 224
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

(train_ds, val_ds), ds_info = tfds.load(
    'imagenet2012',
    split=['train', 'validation'],
    batch_size=64,
    as_supervised=True,
    shuffle_files=True,
    with_info=True,
)
train_ds = train_ds.map(resizer,
    num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(resizer,
    num_parallel_calls=AUTOTUNE)

print(tfds.show_statistics(ds_info), file=sys.stderr)

datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
)
#datagen.fit(train_ds)

model = u_net(
    (IMG_SIZE, IMG_SIZE, N_CHANNELS),
    nb_filters=32,
    conv_size=3,
    initialization="he_normal",
    activation=Swish,
    output_channels=1000,
    batchnorm=True,
    maxpool=False,
    sigma_noise=0,
    arch="U-Net",
    encode_only=True,
)

model.compile(
    keras.optimizers.Adam(1.0e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['acc'],
)

os.makedirs("logs/imagenet32_pretrain/", exist_ok=True)
csv_path = "logs/imagenet32_pretrain/metrics.csv"
csv_logger = CSVLogger(csv_path, append=True)

tb_path = "logs/imagenet32_pretrain/tb"
tb_callback = TensorBoard(
    log_dir=tb_path,
    histogram_freq=0,
    write_graph=True,
    embeddings_freq=0,
    update_freq="epoch",
    profile_batch=0, # disable profiling for now
)

early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=0.0001, patience=10, verbose=0, mode="auto"
)
reduce_lr_on_plateau = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=5, min_lr=1e-8, verbose=1
)
pbar = tfa.callbacks.TQDMProgressBar()
os.makedirs("models/imagenet32_pretrain/", exist_ok=True)
model_checkpoint = PatchedModelCheckpoint(
    "models/imagenet32_pretrain/pretrain_epoch_{epoch}_acc_{acc:0.4f}.h5",
    verbose=0,
    monitor="acc",
    save_best_only=True,
)
history = model.fit(
    x=train_ds,
    validation_data=val_ds,
    epochs=100,
    workers=30,
    verbose=2,
    callbacks=[
        pbar,
        model_checkpoint,
        tb_callback,
        csv_logger,
        early_stopping,
        reduce_lr_on_plateau,
    ],
)
