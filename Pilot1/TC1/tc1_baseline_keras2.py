from __future__ import print_function

import os

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    LocallyConnected1D,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential, model_from_json

file_path = os.path.dirname(os.path.realpath(__file__))

import candle
import tc1 as bmk
import tensorflow as tf
from callback_utils import RecordBatch

def initialize_parameters(default_model="tc1_default_model.txt"):

    # Build benchmark object
    tc1Bmk = bmk.BenchmarkTC1(
        file_path,
        default_model,
        "keras",
        prog="tc1_baseline",
        desc="Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(tc1Bmk)

    return gParameters


def run(gParameters):

    num_gpu = gParameters['num_gpu']
    gpu_type = gParameters['gpu_type']
    save_dir = f'benchmark_logs/{num_gpu}x{gpu_type}'

    X_train, Y_train, X_test, Y_test = bmk.load_data(gParameters)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("Y_train shape:", Y_train.shape)
    print("Y_test shape:", Y_test.shape)

    x_train_len = X_train.shape[1]

    # this reshaping is critical for the Conv1D to work

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        model = Sequential()
        dense_first = True

        layer_list = list(range(0, len(gParameters["conv"]), 3))
        for _, i in enumerate(layer_list):
            filters = gParameters["conv"][i]
            filter_len = gParameters["conv"][i + 1]
            stride = gParameters["conv"][i + 2]
            print(i / 3, filters, filter_len, stride)
            if gParameters["pool"]:
                pool_list = gParameters["pool"]
                if type(pool_list) != list:
                    pool_list = list(pool_list)

            if filters <= 0 or filter_len <= 0 or stride <= 0:
                break
            dense_first = False
            if "locally_connected" in gParameters:
                model.add(
                    LocallyConnected1D(
                        filters,
                        filter_len,
                        strides=stride,
                        padding="valid",
                        input_shape=(x_train_len, 1),
                    )
                )
            else:
                # input layer
                if i == 0:
                    model.add(
                        Conv1D(
                            filters=filters,
                            kernel_size=filter_len,
                            strides=stride,
                            padding="valid",
                            input_shape=(x_train_len, 1),
                        )
                    )
                else:
                    model.add(
                        Conv1D(
                            filters=filters,
                            kernel_size=filter_len,
                            strides=stride,
                            padding="valid",
                        )
                    )
            model.add(Activation(gParameters["activation"]))
            if gParameters["pool"]:
                model.add(MaxPooling1D(pool_size=pool_list[i // 3]))

        if not dense_first:
            model.add(Flatten())

        for i, layer in enumerate(gParameters["dense"]):
            if layer:
                if i == 0 and dense_first:
                    model.add(Dense(layer, input_shape=(x_train_len, 1)))
                else:
                    model.add(Dense(layer))
                model.add(Activation(gParameters["activation"]))
                if gParameters["dropout"]:
                    model.add(Dropout(gParameters["dropout"]))

        if dense_first:
            model.add(Flatten())

        model.add(Dense(gParameters["classes"]))
        model.add(Activation(gParameters["out_activation"]))

        # model.summary()

        model.compile(
            loss=gParameters["loss"],
            optimizer=gParameters["optimizer"],
            metrics=[gParameters["metrics"]],
        )

    output_dir = gParameters["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set up callbacks to do work during model training..
    model_name = gParameters["model_name"]
    path = "{}/{}.autosave.model.h5".format(output_dir, model_name)
    checkpointer = ModelCheckpoint(
        filepath=path, verbose=1, save_weights_only=False, save_best_only=True
    )
    csv_logger = CSVLogger("{}/training.log".format(output_dir))
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=1,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
    )

    custom_callback = RecordBatch(save_dir = save_dir, 
                                    model_name = 'tc1',
                                    end_batch = gParameters['iter_limit'])

    history = model.fit(
        X_train,
        Y_train,
        batch_size=gParameters["batch_size"],
        epochs=gParameters["epochs"],
        verbose=1,
        validation_data=(X_test, Y_test),
        callbacks=[reduce_lr, custom_callback],
    )

    return history


def main():

    gParameters = initialize_parameters()
    run(gParameters)


if __name__ == "__main__":
    main()
    try:
        K.clear_session()
    except AttributeError:  # theano does not have this function
        pass
