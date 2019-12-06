import pkgutil
from pathlib import Path

import click
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from SHMModels.fitted_models import ContextModel
from keras import optimizers, Input, Model

# Path to germline sequence
from build_nns import build_nn
from genDat import *
from genDat import hot_encode_2d, gen_batch
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
    UpSampling2D,
)

##### USER INPUTS (Edit some of these to be CLI eventually)

germline_sequence = "data/gpt.fasta"
# Context model length and pos_mutating
context_model_length = 3
context_model_pos_mutating = 2
# Path to aid model
aid_context_model = "data/aid_logistic_3mer.csv"
# Num seqs and n_mutation rounds
n_seqs = 1
n_mutation_rounds = 10
# step size
step_size = 0.0001
# batch size num epochs
batch_size = 300
num_epochs = 1000
steps_per_epoch = 1
# flag to include ber_pathway
ber_pathway = 1

# Means and sds from set of 5000 prior samples (logit transform 4:8)
means = [
    0.50228154,
    26.8672,
    0.08097563,
    0.07810973,
    -1.52681097,
    -1.49539369,
    -1.49865018,
    -1.48759332,
    0.50265601,
]
sds = [
    0.29112116,
    12.90099082,
    0.1140593,
    0.11241542,
    1.42175933,
    1.43498051,
    1.44336424,
    1.43775417,
    0.28748498,
]

# Load sequence into memory
sequence = list(
    SeqIO.parse(germline_sequence, "fasta", alphabet=IUPAC.unambiguous_dna)
)[0]
# Load aid model into memory
aid_model_string = pkgutil.get_data("SHMModels", aid_context_model)
aid_model = ContextModel(
    context_model_length, context_model_pos_mutating, aid_model_string
)
orig_seq = hot_encode_2d(sequence)
# Create testing data
junk = gen_batch(
    batch_size,
    sequence,
    aid_model,
    n_seqs,
    n_mutation_rounds,
    orig_seq,
    means,
    sds,
    2,
    4,
    ber_pathway,
)
t_batch_data = junk["seqs"][:, 0, :, :, :]
t_batch_labels = np.swapaxes(junk["mechs"][:, 0, :, :], 1, 2)

# Let's build our encoder
input_seq = Input(shape=(308, 4, 1))

x = Conv2D(16, (3, 4), activation="relu", padding="same")(input_seq)
x = MaxPooling2D((2, 1), padding="same")(x)
x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
encoded = MaxPooling2D((2, 2), padding="same")(x)

# Now we decode back up
x = Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 4), activation="relu")(x)
x = UpSampling2D((2, 3))(x)
decoded = Conv2D(1, (3, 3), activation="relu", padding="same")(x)
decoded = Reshape((308, 3))(decoded)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional
autoencoder = Model(input_seq, decoded)
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

print(autoencoder.summary(90))

# Create iterator for simulation
def genTraining(batch_size):
    while True:
        # Get training data for step
        dat = gen_batch(
            batch_size,
            sequence,
            aid_model,
            n_seqs,
            n_mutation_rounds,
            orig_seq,
            means,
            sds,
            2,
            4,
            ber_pathway,
        )
        # We repeat the labels for each x in the sequence
        batch_labels = np.swapaxes(dat["mechs"][:, 0, :, :], 1, 2)
        batch_data = dat["seqs"][:, 0, :, :, :]
        yield batch_data, batch_labels


history = autoencoder.fit_generator(
    genTraining(batch_size),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(t_batch_data, t_batch_labels),
)

print(np.mean(np.square(t_batch_labels)))
print(min(history.history["val_loss"]))

# Save predictions and labels
np.savetxt(Path("sims/", "labels"), t_batch_labels, delimiter=",")
np.savetxt(Path("sims/", "preds"), autoencoder.predict(t_batch_data))
# Save  model loss
np.savetxt(Path("sims/", "loss"), history.history["val_loss"])
autoencoder.save(Path("sims/", "model"))
