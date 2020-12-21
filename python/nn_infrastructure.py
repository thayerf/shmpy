# Load our stuff
import pkgutil
from pathlib import Path
import numpy as np
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from SHMModels.fitted_models import ContextModel
from keras import optimizers, Input, Model
from keras.models import Sequential
import keras.backend as K
import matplotlib.pyplot as plt
from keras.layers import (
    Dense,
    TimeDistributed,
    SimpleRNN,
    Dropout,
    Conv2D,
    Conv2DTranspose,
    Flatten,
    Conv1D,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
)
import scipy
import math

# We want a custom loss that only penalizes GP at lesion values
def custom_loss(y_true, y_pred):
    g_true = y_true[:, :, 0]
    g_pred = y_pred[:, :, 0]
    ind = K.cast((y_true[:, :, 1] + y_true[:, :, 2]) > 0, "float32")
    g_adj = (g_true - g_pred) * ind
    return K.mean(
        K.square(
            K.stack(
                (
                    g_adj,
                    y_true[:, :, 1] - y_pred[:, :, 1],
                    y_true[:, :, 2] - y_pred[:, :, 2],
                )
            )
        )
    )


# Metric which gives mse over GP vec only (and only at places with lesions)
def cond_variance(y_true, y_pred):
    g_true = y_true[:, :, 0]
    g_pred = y_pred[:, :, 0]
    ind = K.cast((y_true[:, :, 1] + y_true[:, :, 2]) > 0, "float32")
    return K.mean(K.square(g_true - g_pred))


# Build NN for sequence on length seq_length
def build_nn(seq_length):
    # Let's build our encoder.
    input_seq = Input(shape=(seq_length, 4, 1))

    # We add 2 convolutional layers.
    x = Conv2D(16, (3, 6), activation="relu", padding="same")(input_seq)
    x = MaxPooling2D((2, 1), padding="same")(x)
    x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # Now we decode back up
    x = Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    )(x)
    x = Conv2DTranspose(
        filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    )(x)
    x = Conv2DTranspose(
        filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu"
    )(x)
    x = Flatten()(x)
    # I think ReLU is fine here because the values are nonnegative?
    decoded = Dense(units=(seq_length * 3), activation="linear")(x)
    reshaped = Reshape((seq_length, 3))(decoded)

    # at this point the "decoded" representation is a 3*seq_length vector indicating our predicted # of
    # lesions, prelesions, gp values at each site.
    autoencoder = Model(input_seq, reshaped)
    return autoencoder
