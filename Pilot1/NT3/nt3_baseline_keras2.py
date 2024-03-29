from __future__ import print_function

import os

import candle
import nt3 as bmk
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau
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
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from callback_utils import RecordBatch

def initialize_parameters(default_model="nt3_default_model.txt"):

    # Build benchmark object
    nt3Bmk = bmk.BenchmarkNT3(
        bmk.file_path,
        default_model,
        "keras",
        prog="nt3_baseline",
        desc="1D CNN to classify RNA sequence data in normal or tumor classes",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(nt3Bmk)

    return gParameters


def load_data(train_path, test_path, gParameters):

    print("Loading data...")
    df_train = (pd.read_csv(train_path, header=None).values).astype("float32")
    df_test = (pd.read_csv(test_path, header=None).values).astype("float32")
    print("done")

    print("df_train shape:", df_train.shape)
    print("df_test shape:", df_test.shape)

    seqlen = df_train.shape[1]

    df_y_train = df_train[:, 0].astype("int")
    df_y_test = df_test[:, 0].astype("int")

    Y_train = to_categorical(df_y_train, gParameters["classes"])
    Y_test = to_categorical(df_y_test, gParameters["classes"])

    df_x_train = df_train[:, 1:seqlen].astype(np.float32)
    df_x_test = df_test[:, 1:seqlen].astype(np.float32)

    X_train = df_x_train
    X_test = df_x_test

    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    mat = scaler.fit_transform(mat)

    X_train = mat[: X_train.shape[0], :]
    X_test = mat[X_train.shape[0] :, :]

    return X_train, Y_train, X_test, Y_test


def run(gParameters):

    file_train = gParameters["train_data"]
    file_test = gParameters["test_data"]
    url = gParameters["data_url"]
    num_gpu = gParameters['num_gpu']
    gpu_type = gParameters['gpu_type']
    save_dir = f'benchmark_logs/{num_gpu}x{gpu_type}'

    train_file = candle.get_file(file_train, url + file_train, cache_subdir="Pilot1")
    test_file = candle.get_file(file_test, url + file_test, cache_subdir="Pilot1")

    initial_epoch = 0
    best_metric_last = None

    X_train, Y_train, X_test, Y_test = load_data(train_file, test_file, gParameters)

    # only training set has noise
    X_train, Y_train = candle.add_noise(X_train, Y_train, gParameters)

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

        layer_list = list(range(0, len(gParameters["conv"]), 3))
        for _, i in enumerate(layer_list):
            filters = gParameters["conv"][i]
            filter_len = gParameters["conv"][i + 1]
            stride = gParameters["conv"][i + 2]
            print(int(i / 3), filters, filter_len, stride)
            if gParameters["pool"]:
                pool_list = gParameters["pool"]
                if type(pool_list) != list:
                    pool_list = list(pool_list)

            if filters <= 0 or filter_len <= 0 or stride <= 0:
                break
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
                model.add(MaxPooling1D(pool_size=pool_list[int(i / 3)]))

        model.add(Flatten())

        for layer in gParameters["dense"]:
            if layer:
                model.add(Dense(layer))
                model.add(Activation(gParameters["activation"]))
                if gParameters["dropout"]:
                    model.add(Dropout(gParameters["dropout"]))
        model.add(Dense(gParameters["classes"]))
        model.add(Activation(gParameters["out_activation"]))

    # J = candle.restart(gParameters, model)
    # if J is not None:
    #     initial_epoch = J["epoch"]
    #     best_metric_last = J["best_metric_last"]
    #     gParameters["ckpt_best_metric_last"] = best_metric_last
    #     print("initial_epoch: %i" % initial_epoch)

    # ckpt = candle.CandleCheckpointCallback(gParameters, verbose=False)
        kerasDefaults = candle.keras_default_config()

        # Define optimizer
        optimizer = candle.build_optimizer(
            gParameters["optimizer"], gParameters["learning_rate"], kerasDefaults
        )

        model.summary()
        model.compile(
            loss=gParameters["loss"], optimizer=optimizer, metrics=[gParameters["metrics"]]
        )

    output_dir = gParameters["output_dir"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # calculate trainable and non-trainable params
    gParameters.update(candle.compute_trainable_params(model))

    # set up a bunch of callbacks to do work during model training..
    model_name = gParameters["model_name"]
    # path = '{}/{}.autosave.model.h5'.format(output_dir, model_name)
    # checkpointer = ModelCheckpoint(filepath=path, verbose=1, save_weights_only=False, save_best_only=True)
    candleRemoteMonitor = candle.CandleRemoteMonitor(params=gParameters)
    timeoutMonitor = candle.TerminateOnTimeOut(gParameters["timeout"])

    custom_callback = RecordBatch(save_dir = save_dir, 
                                    model_name = 'nt3',
                                    end_batch = gParameters['iter_limit'])

    print(gParameters["batch_size"])
    history = model.fit(
        X_train,
        Y_train,
        batch_size=gParameters["batch_size"],
        epochs=gParameters["epochs"],
        initial_epoch=initial_epoch,
        verbose=1,
        # validation_data=(X_test, Y_test),
        callbacks=[candleRemoteMonitor, timeoutMonitor, custom_callback],
    )

    score = model.evaluate(X_test, Y_test, verbose=0)

    if False:
        print("Test score:", score[0])
        print("Test accuracy:", score[1])
        # serialize model to JSON
        model_json = model.to_json()
        with open("{}/{}.model.json".format(output_dir, model_name), "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights("{}/{}.weights.h5".format(output_dir, model_name))
        print("Saved model to disk")

        # load json and create model
        json_file = open("{}/{}.model.json".format(output_dir, model_name), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model_json = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model_json.load_weights(
            "{}/{}.weights.h5".format(output_dir, model_name)
        )
        print("Loaded json model from disk")

        # evaluate json loaded model on test data
        loaded_model_json.compile(
            loss=gParameters["loss"],
            optimizer=gParameters["optimizer"],
            metrics=[gParameters["metrics"]],
        )
        score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

        print("json Test score:", score_json[0])
        print("json Test accuracy:", score_json[1])

        print(
            "json %s: %.2f%%"
            % (loaded_model_json.metrics_names[1], score_json[1] * 100)
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
