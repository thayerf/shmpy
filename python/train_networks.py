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
import click
from pathlib import Path

##### USER INPUTS (Edit some of these to be CLI eventually)

# Path to germline sequence
from build_nns import build_nn
from genDat import hot_encode_2d, gen_batch_1d, gen_batch

germline_sequence = "data/gpt.fasta"
# Context model length and pos_mutating
context_model_length = 3
context_model_pos_mutating = 2
# Path to aid model
aid_context_model = "data/aid_logistic_3mer.csv"
# Num seqs and n_mutation rounds
n_seqs = 50
n_mutation_rounds = 3
# Number of hidden layers
num_hidden = 5
# step size and step decay
step_size = 0.0001
# batch size num epochs
batch_size = 1000
num_epochs = 400
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


# For right now, the only CLI arguments are the type of network and the dimension of the encoding.
@click.command()
@click.argument("network_type")
@click.argument("encoding_type", type=int)
@click.argument("encoding_length", type=int)
@click.argument("output_path")
def main(network_type, encoding_type, encoding_length, output_path):
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
    # model = build_nn(network_type, 9)
    # adam = optimizers.adam(lr=step_size)
    # print(model.summary(90))
    # Create testing data
    junk = gen_batch(
        3, sequence, aid_model, 3, 500, orig_seq, means, sds, 2, 4, ber_pathway,
    )
    t_batch_data = junk["seqs"]
    t_batch_labels = junk["params"]

    # # Create iterator for simulation
    # def genTraining(batch_size):
    #     while True:
    #         # Get training data for step
    #         dat = gen_batch(
    #             batch_size,
    #             sequence,
    #             aid_model,
    #             n_seqs,
    #             n_mutation_rounds,
    #             orig_seq,
    #             means,
    #             sds,
    #             encoding_type,
    #             encoding_length,
    #             ber_pathway,
    #         )
    #         # We repeat the labels for each x in the sequence
    #         batch_labels = dat["params"]
    #         batch_data = dat["seqs"]
    #         yield batch_data, batch_labels
    #
    # # We use MSE for now
    # model.compile(loss="mean_squared_error", optimizer=adam)
    # # Train the model on this epoch
    # history = model.fit_generator(
    #     genTraining(batch_size),
    #     epochs=num_epochs,
    #     steps_per_epoch=steps_per_epoch,
    #     validation_data=(t_batch_data, t_batch_labels),
    # )
    # # Save predictions and labels
    # np.savetxt(Path(output_path, "labels"), t_batch_labels, delimiter=",")
    # np.savetxt(Path(output_path, "preds"), model.predict(t_batch_data))
    # # Save  model loss
    # np.savetxt(Path(output_path, "loss"), history.history["val_loss"])
    # model.save(Path(output_path, "model"))


if __name__ == "__main__":
    main()
