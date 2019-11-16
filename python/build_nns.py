from keras.models import Sequential
from keras.layers import (
    Dense,
    TimeDistributed,
    SimpleRNN,
    Dropout,
    Conv2D,
    Flatten,
    Conv1D,
    MaxPooling2D,
    Reshape,
)


def build_nn(type, output_dim):
    model = Sequential()
    if type == "crnn":
        model = build_crnn(output_dim)
    elif type == "ffrnn":
        model = build_ffrnn(output_dim)
    else:
        raise RuntimeError("Network type must be either 'crnn' or 'ffrnn'")
    return model


def build_crnn(output_length):
    model = Sequential()
    model.add(
        TimeDistributed(
            Conv2D(64, kernel_size=(10, 4), strides=(1, 1), activation="relu"),
            input_shape=(None, 308, 4, 1),
        )
    )
    model.add(TimeDistributed(Reshape((299, 64, 1))))
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(
        TimeDistributed(
            Conv2D(32, kernel_size=(8, 4), strides=(1, 1), activation="relu")
        )
    )
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(16, activation="relu")))
    model.add(SimpleRNN(output_length, activation="linear"))
    return model


def build_ffrnn(encoding_length, output_length):
    model = Sequential()
    model.add(
        TimeDistributed(
            Dense(
                512,
                activation="relu",
                kernel_initializer="zeros",
                bias_initializer="zeros",
            ),
            input_shape=(None, 308 * encoding_length),
        )
    )
    model.add(Dropout(0.2))
    model.add(
        TimeDistributed(
            Dense(
                256,
                activation="relu",
                kernel_initializer="zeros",
                bias_initializer="zeros",
            )
        )
    )
    model.add(Dropout(0.2))
    model.add(
        TimeDistributed(
            Dense(
                128,
                activation="relu",
                kernel_initializer="zeros",
                bias_initializer="zeros",
            )
        )
    )
    model.add(Dropout(0.2))
    model.add(
        TimeDistributed(
            Dense(
                64,
                activation="relu",
                kernel_initializer="zeros",
                bias_initializer="zeros",
            )
        )
    )
    model.add(Dropout(0.2))
    model.add(
        SimpleRNN(
            output_length,
            activation="linear",
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )
    )
    return model
