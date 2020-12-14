import numpy as np
import cox_process_functions as cpf

# Get a 2d hot encoding of a sequence
#' @param seq: Character vector of sequence
def hot_encode_2d(seq):
    seq_hot = np.zeros((len(seq), 4, 1))
    for j in range(len(seq)):
        seq_hot[j, 0, 0] = seq[j] == "A"
        seq_hot[j, 1, 0] = seq[j] == "T"
        seq_hot[j, 2, 0] = seq[j] == "G"
        seq_hot[j, 3, 0] = seq[j] == "C"
    return seq_hot


# Discrete to interval, but for whole vector
def conv_to_short(A_long, A_tilde_long, g_long):
    g = []
    A = []
    A_tilde = []
    for p in range(len(A_tilde_long)):
        if A_long[p] > 0:
            for q in range(int(A_long[p])):
                g = np.append(g, g_long[p])
                A = np.append(A, cpf.discrete_to_interval(p, len(A_long)))
    for p in range(len(A_tilde_long)):
        if A_tilde_long[p] > 0:
            for q in range(int(A_tilde_long[p])):
                g = np.append(g, g_long[p])
                A_tilde = np.append(A_tilde, cpf.discrete_to_interval(p, len(A_long)))
    return A, A_tilde, g


# Generate batch of mutated seqs in nn friendly format
def gen_nn_batch(germline, c_array, params, ber_params, batch_size):
    mut = []
    lat = []
    for i in range(batch_size):
        seq, A_long, A_tilde_long, g_long = cpf.forward_sample_sequences(
            c_array,
            params["base_rate"],
            params["lengthscale"],
            params["gp_sigma"],
            germline,
            ber_params,
            params["gp_ridge"],
            params["gp_offset"],
        )
        mut.append(hot_encode_2d(seq))
        lat.append(np.stack([g_long, A_long, A_tilde_long], axis=1))
    return np.array(mut), np.array(lat)


# Generate batch of mutated seqs in nn friendly format and keep char vec of seqs
def gen_batch_with_seqs(germline, c_array, params, ber_params, batch_size):
    mut = []
    lat = []
    seqs = []
    for i in range(batch_size):
        seq, A_long, A_tilde_long, g_long = cpf.forward_sample_sequences(
            c_array,
            params["base_rate"],
            params["lengthscale"],
            params["gp_sigma"],
            germline,
            ber_params,
            params["gp_ridge"],
            params["gp_offset"],
        )
        mut.append(hot_encode_2d(seq))
        lat.append(np.stack([g_long, A_long, A_tilde_long], axis=1))
        seqs.append(seq)
    return np.array(mut), np.array(lat), seqs
