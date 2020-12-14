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
#' @param lambda_star Mean number of prelesions per site
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
    return base


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
    output_seq = gl_seq
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


#' @param seq: A character vector with the mutated sequence
#' @param gl_seq A character vector with the germline sequence
#' @param A: A vector, with the locations of the lesions
#' @param A_tilde: A vector, with the locations of the prelesions
#' @param g: A vector with the values of the GP at [A, A_tilde]
#' @param lengthscale: The lengthscale for the GP
#' @param sigma: The scale for the GP
#' @param gp_ridge: Ridge for the GP
#' @return: Log probability of the complete data for the SGCP
def sgcp_complete_data_log_prob(
    seq,
    gl_seq,
    A,
    A_tilde,
    g,
    lambda_star,
    lengthscale,
    sigma,
    gp_ridge,
    gp_offset,
    ber_params,
):
    K = len(A)
    M = len(A_tilde)
    # Since we only count lesions and prelesions at C sites, but do not sample them only at C sites, we have to adjust the Poisson mean for the likelihood.
    c_mean = np.mean([x == "C" for x in gl_seq])
    lam_adj = lambda_star * c_mean
    # P( A_tilde + A )
    pois = (K + M) * np.log(lam_adj) - lam_adj - math.lgamma(K + M + 1)
    # P(A, A_tilde | g)
    thinning = thinning_log_prob(g, K, M)
    gp_kern_ridged = make_se_kernel(
        np.append(np.array(A), A_tilde), lengthscale, sigma, gp_ridge
    )
    # P(g)
    gp = np.log(
        scipy.stats.multivariate_normal.pdf(
            g, mean=gp_offset + np.zeros(len(g)), cov=gp_kern_ridged
        )
    )
    return (pois + thinning + gp, pois, thinning, gp)


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


#' @param complete_data: A dictionary containing the estimated lesions (A),
#' prelesinons (A_tilde), and values of the GP at the lesions/prelesions (g)
#' @param current_model_params: A dict with the current model parameters.
#' @param sampling_sd: Instead of the true GP values in g, we use g +
#' gp noise, sampling_sd is the sd of that noise (cond_var output by NN)
#' @return: A dict, with the complete data (lesions = A, prelesions =
#' A_tilde, GP values = g) and the weight for that sample (w)
def complete_data_sample_around_cond_means_sgcp(
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
#' @param A: The lesions
#' @param A_tilde: The prelesions
#' @param g: The values of the GP at [A, A_tilde]
#' @param model_params: A dict with the current model params
#' @return The complete-data log probability of seq, A, A_tilde, g given the current model parameters.
def sequence_complete_data_log_prob(seq, gl_seq, A, A_tilde, g, model_params):
    lambda_star = model_params["lambda_star"]
    lengthscale = model_params["lengthscale"]
    sigma = model_params["sigma"]
    gp_ridge = model_params["gp_ridge"]
    ber_params = model_params["ber_params"]
    sgcp_log_prob = sgcp_complete_data_log_prob(
        A, A_tilde, g, lambda_star, lengthscale, sigma, gp_ridge
    )
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
            return -Inf
    log_p = 0
    for i in A_indices:
        log_p = log_p + np.log(ber_params[seq[i]])
    return log_p
