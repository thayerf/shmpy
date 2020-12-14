# Load our stuff
import numpy as np
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from nn_utils import gen_nn_batch
from nn_infrastructure import build_nn, custom_loss, cond_variance
from keras import optimizers

# Path to germline sequence
germline_sequence = "data/gpt.fasta"
# Context model length and pos_mutating
context_model_length = 3
context_model_pos_mutating = 2
# Path to aid model
aid_context_model = "data/aid_logistic_3mer.csv"
batch_size = 10
num_epochs = 100
steps_per_epoch = 1
step_size = 0.1
germline = list(
    list(SeqIO.parse(germline_sequence, "fasta", alphabet=IUPAC.unambiguous_dna))[0].seq
)

n = np.size(germline)
c_array = np.zeros(n)
for i in range(n):
    c_array[i] = 1.0 * (germline[i] == "C")

start_model_params = {
    "base_rate": 300.0,
    "lengthscale": 0.05,
    "gp_sigma": 10.0,
    "gp_ridge": 0.05,
    "gp_offset": -10,
}
ber_params = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}

opt = optimizers.Adam(learning_rate=step_size)
autoencoder = build_nn(308)
autoencoder.compile(optimizer=opt, loss=custom_loss, metrics=[cond_variance])


def genTraining(batch_size):
    while True:
        # Get training data for step
        mut, les = gen_nn_batch(
            germline, c_array, start_model_params, ber_params, batch_size
        )
        yield mut, les


t_batch_data, t_batch_labels = gen_nn_batch(
    germline, c_array, start_model_params, ber_params, 10
)

history = autoencoder.fit_generator(
    genTraining(batch_size),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(t_batch_data, t_batch_labels),
    verbose=2,
)
