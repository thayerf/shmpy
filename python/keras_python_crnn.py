import numpy as np
import numpy.random
import pkgutil
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from SHMModels.summary_statistics import write_all_stats
from SHMModels.fitted_models import ContextModel
from SHMModels.simulate_mutations import memory_simulator
import time
from scipy.special import logit
from genDat import *
from keras.models import Sequential
from keras.layers import (
    LSTM,
    Dense,
    TimeDistributed,
    SimpleRNN,
    Input,
    Dropout,
    Conv2D,
    ConvLSTM2D,
    Conv3D,
    BatchNormalization,
    Flatten,
    Conv1D,
    MaxPooling2D,
    Reshape,
    Activation,
)
from keras import optimizers
from sklearn.preprocessing import scale
import warnings
import matplotlib.pyplot as plt
import click

warnings.filterwarnings(action="ignore")

### USER INPUTS (Non-click are fixed)
# Path to germline sequence
germline_sequence = "data/gpt.fasta"
# Context model length and pos_mutating
context_model_length = 3
context_model_pos_mutating = 2
# Path to aid model
aid_context_model = "data/aid_logistic_3mer.csv"
# Num seqs and n_mutation rounds
n_seqs = 50
n_mutation_rounds = 3
# step size and step decay
step_size = 0.00001
# batch size num epochs
batch_size = 1000
num_epochs = 200
steps_per_epoch = 1


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
##### END USER INPUTS


# Load sequence into memory
sequence = list(
    SeqIO.parse(germline_sequence, "fasta", alphabet=IUPAC.unambiguous_dna)
)[0]
# Load aid model into memory
aid_model_string = pkgutil.get_data("SHMModels", aid_context_model)
aid_model = ContextModel(
    context_model_length, context_model_pos_mutating, aid_model_string
)
orig_seq = hot_encode_2d(str(sequence.seq))

# Create iterator for simulation
def genTraining_2d(batch_size):
    while True:
        # Get training data for step
        dat = gen_batch_2d(
            batch_size,
            sequence,
            aid_model,
            n_seqs,
            n_mutation_rounds,
            orig_seq,
            means,
            sds,
        )
        # We repeat the labels for each x in the sequence
        batch_labels = dat["params"]
        batch_data = dat["seqs"]
        yield batch_data, batch_labels


# Create testing data
junk = gen_batch_2d(
    500, sequence, aid_model, n_seqs, n_mutation_rounds, orig_seq, means, sds
)
t_batch_data = junk["seqs"]
t_batch_labels = junk["params"]


# Create Network
model = Sequential()
model.add(TimeDistributed(Dense(64, activation="relu"), input_shape=(None, 308, 9, 1)))
model.add(TimeDistributed(Dense(32, activation="relu")))
model.add(TimeDistributed(Dense(16, activation="relu")))
model.add(SimpleRNN(9, activation="linear"))


# Print model summary
print(model.summary(90))
adam = optimizers.adam(lr=step_size)
model.compile(loss="mean_squared_error", optimizer=adam)
# Train the model on this epoch
history = model.fit_generator(
    genTraining_2d(batch_size),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(t_batch_data, t_batch_labels),
    verbose=2,
)

# Save predictions and labels
np.savetxt("sims/crnn/labels", t_batch_labels, delimiter=",")
np.savetxt("sims/crnn/preds", model.predict(t_batch_data))
# Save  model loss
np.savetxt("sims/crnn/loss", history.history["val_loss"])
model.save("sims/crnn/shmr_crnn_model")
