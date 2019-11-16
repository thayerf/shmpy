from SHMModels.simulate_mutations import simulate_sequences_abc
from SHMModels.simulate_mutations import memory_simulator

# We would like to simulate sequences in place, with the fasta file and context model loaded into memory
# This way we can simulate A LOT of training data without writing to disk (slow)
import numpy as np
from scipy.special import logit
from sklearn.preprocessing import scale

# Get a 2d hot encoding of a sequence
def hot_encode_2d(seq):
    seq_hot = np.zeros((len(seq), 4, 1))
    for j in range(len(seq)):
        seq_hot[j, 0, 0] = seq[j] == "A"
        seq_hot[j, 1, 0] = seq[j] == "T"
        seq_hot[j, 2, 0] = seq[j] == "G"
        seq_hot[j, 3, 0] = seq[j] == "C"
    return seq_hot


# Get a 1d hot encoding of a sequence
def hot_encode_1d(seq):
    seq_hot = np.zeros((len(seq) * 4))
    for j in range(len(seq)):
        seq_hot[4 * j] = seq[j] == "A"
        seq_hot[4 * j + 1] = seq[j] == "T"
        seq_hot[4 * j + 2] = seq[j] == "G"
        seq_hot[4 * j + 3] = seq[j] == "C"
    return seq_hot


# Generate a batch of mutated sequences with given encoding
def gen_batch(
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
):
    params, seqs, mech = memory_simulator(
        sequence, aid_model, n_seqs, n_mutation_rounds, batch_size, ber_pathway
    )
    seqs = seqs[:, 0]
    seqs = [i.decode("utf-8") for i in seqs]
    seqs = [list(i) for i in seqs]
    # PUT 1 DIM ENCODING INFO HERE
    if encoding_type == 1:
        None
    elif encoding_type == 2:
        if encoding_length == 1:
            raise RuntimeError("Can't 1d encode 2d network input")
        else:
            seqs_hot = np.zeros(
                (len(seqs) // n_seqs, n_seqs, len(seqs[1]), encoding_length, 1)
            )
            for i in range(len(seqs)):
                seqs_hot[i // n_seqs, i % n_seqs, :, 0:4, :] = hot_encode_2d(seqs[i])
            if encoding_length == 9:
                orig_seq_rep = np.repeat(orig_seq, n_seqs * batch_size, axis=2)
                orig_seq_rep = np.moveaxis(orig_seq_rep, -1, 0)
                orig_seq_rep = orig_seq_rep.reshape((batch_size, n_seqs, 308, 4, 1))
                seqs_hot[:, :, :, 4:8, :] = orig_seq_rep
                seqs_hot[:, :, :, 8, :] = (
                    seqs_hot[:, :, :, 0:4, :] == seqs_hot[:, :, :, 4:8, :]
                ).all(axis=3)
            elif encoding_length != 4:
                raise RuntimeError(
                    "Not a valid encoding dimension (must be 1, 4, or 9)"
                )
    if ber_pathway:
        params[:, 4:8] = logit(params[:, 4:8])

    return {
        "seqs": seqs_hot,
        "params": params,
        "mechs": mech.reshape((batch_size, n_seqs, 3, len(orig_seq))),
    }


def gen_batch_1d(
    batch_size,
    sequence,
    aid_model,
    n_seqs,
    n_mutation_rounds,
    orig_seq,
    means,
    sds,
    encoding,
    ber_pathway,
):
    params, seqs = memory_simulator(
        sequence, aid_model, n_seqs, n_mutation_rounds, batch_size, ber_pathway
    )
    seqs = seqs[:, 0]
    seqs = [i.decode("utf-8") for i in seqs]
    seqs = [list(i) for i in seqs]
    # 0 is 9 dim encoding
    if encoding == 0:
        seqs_hot = np.zeros((len(seqs) // n_seqs, n_seqs, len(seqs[1]) * 9))
        for i in range(len(seqs)):
            seqs_hot[i // n_seqs, i % n_seqs, 0:4, :] = hot_encode_1d(seqs[i])
        #### ENCODE SEQ AND INDICATOR HERE
    # 1 is 4 dim encoding
    if encoding == 1:
        seqs_hot = np.zeros((len(seqs) // n_seqs, n_seqs, len(seqs[1]) * 4))
        for i in range(len(seqs)):
            seqs_hot[i // n_seqs, i % n_seqs, :] = hot_encode_1d(seqs[i])

    if encoding == 2:
        np.zeros((len(seqs) // n_seqs, n_seqs, len(seqs[1])))
        ##### ENCODE INDICATOR HERE
    if ber_pathway:
        params[:, 4:8] = logit(params[:, 4:8])
    params = (params - means) / sds
    return {"seqs": seqs_hot, "params": params}
