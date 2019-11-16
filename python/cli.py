#!/opt/conda/envs/shmpy/bin/python
import subprocess
import sys
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

#germline_sequence = "data/gpt.fasta"
# Context model length and pos_mutating
context_model_length = 3
context_model_pos_mutating = 2
# Path to aid model
#aid_context_model = "data/aid_logistic_3mer.csv"
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

def run_cmds(cmds, shell=False):
    for cmd in cmds:
        subprocess.check_call(cmd, shell=shell)


def dry_run_cmds(cmds, outfiles):
    run_cmds(["touch {}".format(outfname) for outfname in outfiles], shell=True)
    for cmd in cmds:
        click.echo(cmd)

def dry_check_run_cmds(cmds, ctx, outfiles=[], shell=False):
    if ctx.obj["DRY"]:
        dry_run_cmds(cmds, outfiles)
    else:
        run_cmds(cmds, shell=shell)

@click.group()
@click.option(
    "--dry/--no-dry",
    default=False,
    help="Do not run, just output command args into output file..",
)
@click.pass_context
def cli(ctx, dry):
    """
    Top level command line interface including top level options such as --dry
    """
    # we explicitly pass --dry to each command (using @click.pass_context) via a click context. See https://click.palletsprojects.com/en/7.x/complex/ for more on context.
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called by means other than in the below if __name__ == '__main__':
    ctx.ensure_object(dict)
    ctx.obj["DRY"] = dry  # set 'DRY' key in the context to the top level option value

# TODO make thayer's command take this form
@cli.command()
@click.option("--option1")
@click.option("--option2")
@click.option("--option3")
@click.pass_context
def command2(ctx, option1, option2, option3):
    """
    A command belonging to the top level cli group that runs native python code as opposed to running external programs with subprocess (see command1).
    """

    def run_some_python_code(option1, option2, option3, outfile):
        """
        This function would be the main code that gets run for command2.
        We are just writing the options to the outfile to simulate creating some output from the command.
        """
        with open(outfile, "w") as fh:
            fh.write("{} {} {}".format(option1, option2, option3))

    outfiles = [
        "{}.{}.{}.{}.txt".format(output, option1, option2, option3)
        for output in ("output1", "output2", "output3")
    ]
    if ctx.obj["DRY"]:
        run_cmds(["touch {}".format(outfname) for outfname in outfiles], shell=True)
        print(" ".join(sys.argv))
    else:
        for outfile in outfiles:
            run_some_python_code(option1, option2, option3, outfile)

# For right now, the only CLI arguments are the type of network and the dimension of the encoding.
@cli.command()
@click.argument("germline_sequence")
@click.argument("aid_context_model")
@click.argument("network_type")
@click.argument("encoding_type", type=int)
@click.argument("encoding_length", type=int)
@click.argument("output_path")
@click.pass_context
def train(ctx, germline_sequence, aid_context_model, network_type, encoding_type, encoding_length, output_path):
    """
    Train the network(s).
    """
    outfiles = {"labels": Path(output_path, "labels"),
                "preds": Path(output_path, "preds"),
                "loss": Path(output_path, "loss"),
                "model": Path(output_path, "model")}
    if ctx.obj["DRY"]:
        run_cmds(["touch {}".format(outfname) for outfname in outfiles.values()], shell=True)
        print(" ".join(sys.argv))
    else:
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
        model = build_nn(network_type, encoding_type, 9)
        adam = optimizers.adam(lr=step_size)
        print(model.summary(90))
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
            encoding_type,
            encoding_length,
            ber_pathway,
        )
        t_batch_data = junk["seqs"]
        t_batch_labels = junk["params"]

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
                    encoding_type,
                    encoding_length,
                    ber_pathway,
                )
                # We repeat the labels for each x in the sequence
                batch_labels = dat["params"]
                batch_data = dat["seqs"]
                yield batch_data, batch_labels

        # We use MSE for now
        model.compile(loss="mean_squared_error", optimizer=adam)
        # Train the model on this epoch
        history = model.fit_generator(
            genTraining(batch_size),
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=(t_batch_data, t_batch_labels),
        )
        # Save predictions and labels
        np.savetxt(outfiles["labels"], t_batch_labels, delimiter=",")
        np.savetxt(outfiles["preds"], model.predict(t_batch_data))
        # Save  model loss
        np.savetxt(outfiles["loss"], history.history["val_loss"])
        model.save(outfiles["model"])


if __name__ == "__main__":
    cli(obj={})
