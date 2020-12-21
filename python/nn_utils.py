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


# Generate batch of mutated seqs in nn friendly format
#' @param germline: Germline sequence
#' @param c_array: Array of c indicators from germline
#' @param params: Model parameter specification dict
#' @param ber_params: Transition probabilities for BER
#' @batch_size: Number of sequences to sample
#' Returns hot encoded mutated sequence and latent state labels
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
#' @param germline: Germline sequence
#' @param c_array: Array of c indicators from germline
#' @param params: Model parameter specification dict
#' @param ber_params: Transition probabilities for BER
#' @batch_size: Number of sequences to sample
#' Returns hot encoded mutated sequence and latent state labels, as well as seq char vecs
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
