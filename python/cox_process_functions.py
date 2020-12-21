import numpy as np
import math
import scipy.stats

#' Make SE kernel at a set of points
#' @param x The set of points for kernel evaluation
#' @param lengthscale Lengthscale in GP
#' @param sigma Overall variance in GP
#' @param gp_ridge The ridge value
def make_se_kernel(x, lengthscale, sigma, gp_ridge):
    D = np.zeros([len(x), len(x)])
    upper_tri = np.triu_indices(len(x), 1)
    D[upper_tri] = (np.array(x)[upper_tri[0]] - np.array(x)[upper_tri[1]]) ** 2
    D += D.T
    K = sigma ** 2 * np.exp(-D / (2 * lengthscale))
    np.fill_diagonal(K, K.diagonal() + gp_ridge)
    return K


#' Samples from a SGCP on [0,1]
#'
#' @param lambda_star Mean number of prelesions
#' @param lengthscale The lengthscale parameter for the SE kernel in
#'     the GP
#' @param gp_sigma The variance parameter for the SE kernel in the GP
#' @param gp_ridge The ridge value
#' @param gp_offset The mean of the GP
#' Returns A dict with A = lesions, A_tilde = prelesions, g =
#' values of the GP on [A, A_tilde], in that order.
def forward_sample_sgcp(lambda_star, lengthscale, gp_sigma, gp_ridge, gp_offset):
    N = np.random.poisson(lam=lambda_star)
    x = np.random.uniform(low=0, high=1, size=N)
    K = make_se_kernel(x, lengthscale, gp_sigma, gp_ridge)
    lambda_of_x = np.random.multivariate_normal(mean=gp_offset + np.zeros(N), cov=K)
    sigma_lambda_of_x = 1 / (1 + np.exp(-lambda_of_x))
    uniforms = np.random.uniform(low=0, high=1, size=len(x))
    A_and_g = [
        (xi, li)
        for (xi, si, li, u) in zip(x, sigma_lambda_of_x, lambda_of_x, uniforms)
        if u < si
    ]
    A_tilde_and_g = [
        (xi, li)
        for (xi, si, li, u) in zip(x, sigma_lambda_of_x, lambda_of_x, uniforms)
        if u >= si
    ]
    A = [a for (a, g) in A_and_g]
    A_tilde = [at for (at, g) in A_tilde_and_g]
    g = [g for (a, g) in A_and_g + A_tilde_and_g]
    # Return esions, prelesions, gp (unit interval)
    return A, A_tilde, g


#' Sample one position from BER distribution
#' @param ber_params A dictionary giving the probabilities of A, T, C, G
#' @return A character, either A, T, C, or G.
def sample_ber(ber_params):
    rm = np.random.multinomial(n=1, pvals=[*ber_params.values()])
    base = [b for (b, m) in zip(ber_params.keys(), rm) if m == 1]
    return base[0]


def interval_to_discrete(a, seq_length):
    return int(np.floor(a * seq_length))


def discrete_to_interval(i, seq_length):
    return (i - np.random.uniform(low=0, high=1, size=1)) / seq_length


#' @param c_array vector of indicator of C base at a site in germline sequence
#' @param lambda_star: Rate for the unthinned poisson process
#' @param lengthscale: Lengthscale parameter for the GP
#' @param gp_sigma: Scale for the GP
#' @param gl_seq: A character vector with the germline sequence
#' @param ber_params: A dictionary giving the probabilities of each base substitution for ber
#' @param gp_ridge: Ridge for the GP
#' @param gp_offset The mean of the GP
#' Returns mutated sequence, Lesions (only at C sites), Prelesions (only at C sites), gp values for these lesions and prelesions
def forward_sample_sequences(
    c_array, lambda_star, lengthscale, gp_sigma, gl_seq, ber_params, gp_ridge, gp_offset
):
    # DEEP copy gl seq to edit
    output_seq = gl_seq[:]
    # AID lesions, continuous
    A, A_tilde, g = forward_sample_sgcp(
        lambda_star, lengthscale, gp_sigma, gp_ridge, gp_offset
    )
    A_long = np.zeros(len(gl_seq))
    A_tilde_long = np.zeros(len(gl_seq))
    g_long = np.zeros(len(gl_seq))
    for j in range(len(A)):
        i = interval_to_discrete(A[j], seq_length=len(gl_seq))
        if c_array[i]:
            output_seq[i] = sample_ber(ber_params)
            A_long[i] += 1
            g_long[i] = g[j]
    for j in range(len(A_tilde)):
        i = interval_to_discrete(A_tilde[j], seq_length=len(gl_seq))
        if c_array[i]:
            A_tilde_long[i] += 1
            g_long[i] = g[j + len(A)]

    return output_seq, A_long, A_tilde_long, g_long


#' @param g: The values of the GP, the first K corresponding to lesions, last M corresponding to prelesions
#' @param K: The number of lesions
#' @param M: The number of prelesions
#' @return: Log probability of thinning out the last M potential
#' lesions and keeping the first K, given values of the GP given in g
def thinning_log_prob(g, K, M):
    if K == 0 and M == 0:
        return 0
    elif K == 0 and M > 0:
        return sum(np.log(logistic(-g)))
    elif K > 0 and M == 0:
        return sum(np.log(logistic(g)))
    else:
        return sum(np.log(logistic(g[range(K)]))) + sum(
            np.log(logistic(-g[range(K, K + M)]))
        )


#' @param A: A vector, with the locations of the lesions
#' @param A_tilde: A vector, with the locations of the prelesions
#' @param g: A vector with the values of the GP at [A, A_tilde]
#' @param lengthscale: The lengthscale for the GP
#' @param sigma: The scale for the GP
#' @param gp_ridge: Ridge for the GP
#' @return: Log probability of the complete data for the SGCP
def sgcp_complete_data_log_prob(
    A, A_tilde, g, lambda_star, lengthscale, sigma, gp_ridge, gp_offset,
):
    K = len(A)
    M = len(A_tilde)

    # P(A_k^{*i}) in weights equation
    pois = (K + M) * np.log(lambda_star) - lambda_star - math.lgamma(K + M + 1)

    # P(A_k^i | A_k^{*i} G_k^i) in weights equation
    thinning = thinning_log_prob(g, K, M)

    gp_kern_ridged = make_se_kernel(
        np.append(np.array(A), A_tilde), lengthscale, sigma, gp_ridge
    )
    # P(G_k^i) in weights equation
    gp = np.log(
        scipy.stats.multivariate_normal.pdf(
            g, mean=gp_offset + np.zeros(len(g)), cov=gp_kern_ridged
        )
    )
    return pois + thinning + gp


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


#' @param x_list: A list, each element corresponding to the lesions + prelesions in one sample
#' @param g_list: A list, each element corresponding to the values of the GP in one sample
#' @param weights: A list, each element corresponding to the weight for one sample
#' @param l_test_grid: The lengthscale points at which to evaluate the likelihood
#' @param model_params: The model parameters
def lengthscale_inference(x_list, g_list, weights, l_test_grid, model_params):
    log_probs = np.zeros(len(l_test_grid))
    for (ls, ls_idx) in zip(l_test_grid, range(len(l_test_grid))):
        g_log_p_at_test = 0.0
        for i in range(len(x_list)):
            g = g_list[i]
            K = make_se_kernel(
                x_list[i],
                lengthscale=ls,
                sigma=model_params["gp_sigma"],
                gp_ridge=model_params["gp_ridge"],
            )
            print(K)
            g_log_p_at_test += weights[i] * np.log(
                scipy.stats.multivariate_normal.pdf(g, cov=K)
            )
        log_probs[ls_idx] = g_log_p_at_test
    return l_test_grid[np.argmax(log_probs)]


#' @param complete_data: A dictionary containing the true lesions (A),
#' prelesinons (A_tilde), and values of the GP at the lesions/prelesions (g)
#' @param current_model_params: A dict with the current model parameters.
#' @param sampling_sd: Instead of the true GP values in g, we use g +
#' multivariate normal noise, sampling_sd is the sd of that noise
#' @return: A dict, with the complete data (lesions = A, prelesions =
#' A_tilde, GP values = g) and the weight for that sample (w)
def complete_data_sample_around_true_states_sgcp(
    complete_data, current_model_params, sampling_sd=0.1
):
    ## each element of observed_data is a list with A (the thinned
    ## points) lambda (the gp at the unthinned points), and x (the
    ## unthinned points).
    A = complete_data["A"]
    A_tilde = complete_data["A_tilde"]
    g = complete_data["g"]
    g_sample = g + np.random.normal(size=len(g), loc=0, scale=sampling_sd)
    is_log_prob = sum(
        np.log(scipy.stats.norm.pdf(g_sample - g, loc=0, scale=sampling_sd))
    )
    complete_data_log_prob = sgcp_complete_data_log_prob(
        A,
        A_tilde,
        g_sample,
        current_model_params["lambda_star"],
        current_model_params["lengthscale"],
        current_model_params["gp_sigma"],
        current_model_params["gp_ridge"],
    )
    log_w = complete_data_log_prob - is_log_prob
    return {"A": A, "A_tilde": A_tilde, "g": g_sample, "w": np.exp(log_w)}


#' @param seq: The observed sequence
#' @param gl_seq: The germline sequence
#' @param c_array: An indicator of c locations in germline
#' @param complete_data: A dictionary containing the estimated lesions (A),
#' prelesinons (A_tilde), and values of the GP at the lesions/prelesions (g)
#' @param params: A dict with the model parameters from which to sample.
#' @param sampling_sd: Instead of the true GP values in g, we use g +
#' gp noise, sampling_sd is the sd of that noise (cond_var output by NN)
#' @return: A dict, with the complete data (lesions = A, prelesions =
#' A_tilde, GP values = g) and the weight for that sample (w)
def complete_data_sample_around_cond_means_sgcp(
    c_array, complete_data, params, ber_params, sampling_sd=0.1
):
    # Define the sequences
    gl_seq = complete_data["S"]
    seq = complete_data["S_1"]

    # Define the conditional means from "complete data"
    # Lesions mean equation (only count c sits and threshold by 0)
    A_mean = np.multiply(np.maximum(complete_data["A"], 0.0), c_array)
    # Prelesions mean equation (minus lesions mean equation)
    A_tilde_mean = np.multiply(np.maximum(complete_data["A_tilde"], 0.0), c_array)
    # GP mean equation
    g = complete_data["g"]

    # Sample lesions and prelesions from mean vectors

    # q_2 sampling in weights equation
    A_long = sample_pois_mean(c_array, A_mean)
    # q_1 sampling in weights equation
    A_tilde_long = sample_pois_mean(c_array, A_tilde_mean)

    # Form interval vectors from discrete vectors
    A, A_tilde, g_mean = conv_to_short(A_long, A_tilde_long, g)

    # Form kernel and sample g

    # q_3 sampling in weights equation
    K = make_se_kernel(
        np.append(np.array(A), np.array(A_tilde)),
        params["lengthscale"],
        sampling_sd,
        params["gp_ridge"],
    )
    if len(g_mean) > 0:
        lambda_of_x = np.random.multivariate_normal(mean=np.zeros(len(g_mean)), cov=K)
        g_sample = g_mean + lambda_of_x
    else:
        g_sample = []
    ## Get importance sample likelihoods
    # For the GP
    # q_3 likelihood from weights equation
    if len(g_mean) > 0:
        # GP with mean vector equal to NN output.
        g_is_log_prob = scipy.stats.multivariate_normal.logpdf(
            g_sample - g_mean, mean=np.zeros(len(g_sample)), cov=K
        )
    else:
        g_is_log_prob = 0.0

    # For the pois

    # This term is the sum of pois w/ diff intensities
    q2_log_prob = np.sum(scipy.stats.poisson.logpmf(k=A_long, mu=A_mean))
    q1_log_prob = np.sum(scipy.stats.poisson.logpmf(k=A_tilde_long, mu=A_tilde_mean))
    pois_is_log_prob = q2_log_prob + q1_log_prob

    # Sum logs of q1,q2,q3
    is_log_prob = g_is_log_prob + pois_is_log_prob

    # This calculates log prob assuming A, A_tilde, g are  latent states (numerator of weights equation)
    complete_data_log_prob = sequence_complete_data_log_prob(
        seq, gl_seq, A, A_tilde, g_sample, params, ber_params
    )

    log_w = complete_data_log_prob - is_log_prob
    return {"A": A, "A_tilde": A_tilde, "g": g_sample, "w": np.exp(log_w)}


#' @param seq: The observed sequence
#' @param gl_seq: The germline sequence
#' @param A: The lesions
#' @param A_tilde: The prelesions
#' @param g: The values of the GP at [A, A_tilde]
#' @param model_params: A dict with the current model params
#' @return The complete-data log probability of seq, A, A_tilde, g given the current model parameters.
def sequence_complete_data_log_prob(
    seq, gl_seq, A, A_tilde, g, model_params, ber_params
):
    # Adjust lambda_star to only include fraction of unit interval corresponding to C sites
    lambda_star = model_params["base_rate"] * np.mean([i == "C" for i in gl_seq])
    lengthscale = model_params["lengthscale"]
    sigma = model_params["gp_sigma"]
    gp_ridge = model_params["gp_ridge"]
    gp_offset = model_params["gp_offset"]

    # Calculates last 3 terms in numerator in weights equation
    sgcp_log_prob = sgcp_complete_data_log_prob(
        A, A_tilde, g, lambda_star, lengthscale, sigma, gp_ridge, gp_offset
    )
    # Calculates P(S^i | A_k^i)
    sequence_given_sgcp_log_prob = sequence_log_prob_given_lesions(
        seq, gl_seq, A, ber_params
    )
    return sgcp_log_prob + sequence_given_sgcp_log_prob


#' @param seq: The observed sequence
#' @param gl_seq: The germline sequence
#' @param A: The lesions
#' @param ber_params: A dict with the substitution probabilities for BER
#' @return The log probability of seq given gl_seq, A, and the BER parameters
def sequence_log_prob_given_lesions(seq, gl_seq, A, ber_params):
    A_indices = [interval_to_discrete(a, len(seq)) for a in A]
    ## if mutation but no lesion at any position, log prob is -Inf
    for i in range(len(seq)):
        if (seq[i] != gl_seq[i]) and not np.isin(i, A_indices):
            return -math.inf
    log_p = 0
    for i in A_indices:
        log_p = log_p + np.log(ber_params[seq[i]])
    return log_p


#' @param A: Poisson mean vector for per-site importance sampling
#' @return Newly sampled A in discrete format
def sample_pois_mean(A):
    # Sample from new A and only count lesions at C sites
    new_A = np.random.poisson(lam=A)
    return new_A


# Discrete to interval, but for whole vector
def conv_to_short(A_long, A_tilde_long, g_long):
    g = []
    A = []
    A_tilde = []
    for p in range(len(A_tilde_long)):
        if A_long[p] > 0:
            for q in range(int(A_long[p])):
                g = np.append(g, g_long[p])
                A = np.append(A, discrete_to_interval(p, len(A_long)))
    for p in range(len(A_tilde_long)):
        if A_tilde_long[p] > 0:
            for q in range(int(A_tilde_long[p])):
                g = np.append(g, g_long[p])
                A_tilde = np.append(A_tilde, discrete_to_interval(p, len(A_long)))
    return A, A_tilde, g
