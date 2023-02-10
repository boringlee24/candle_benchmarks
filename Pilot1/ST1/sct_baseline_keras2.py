# Setup

import os

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

file_path = os.path.dirname(os.path.realpath(__file__))

import candle
import smiles_transformer as st
import tensorflow as tf
from callback_utils import RecordBatch

def initialize_parameters(default_model="class_default_model.txt"):

    # Build benchmark object
    sctBmk = st.BenchmarkST(
        st.file_path,
        default_model,
        "keras",
        prog="sct_baseline",
        desc="Transformer model for SMILES classification",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(sctBmk)

    return gParameters


# Train and Evaluate


def run(params):

    num_gpu = params['num_gpu']
    gpu_type = params['gpu_type']
    save_dir = f'benchmark_logs/{num_gpu}x{gpu_type}'

    x_train, y_train, x_val, y_val = st.load_data(params)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        model = st.transformer_model(params)

        kerasDefaults = candle.keras_default_config()

        optimizer = candle.build_optimizer(
            params["optimizer"], params["learning_rate"], kerasDefaults
        )

        model.compile(loss=params["loss"], optimizer=optimizer, metrics=["accuracy"])

    # set up a bunch of callbacks to do work during model training..

    checkpointer = ModelCheckpoint(
        filepath="smile_class.autosave.model.h5",
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )
    csv_logger = CSVLogger("smile_class.training.log")
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.75,
        patience=20,
        verbose=1,
        mode="auto",
        epsilon=0.0001,
        cooldown=3,
        min_lr=0.000000001,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=100, verbose=1, mode="auto")

    custom_callback = RecordBatch(save_dir = save_dir, 
                                    model_name = 'st1',
                                    end_batch = params['iter_limit'])

    history = model.fit(
        x_train,
        y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        verbose=1,
        callbacks=[reduce_lr, custom_callback],
    )

    # model.load_weights("smile_class.autosave.model.h5")

    return history


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()
