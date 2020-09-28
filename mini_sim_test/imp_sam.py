# Load our stuff

import numpy as np
import scipy
import math
from minisim import *

def complete_data_sample_around_true_states_sgcp(complete_data,
                                                 params,
                                                 sampling_sd = .1):
    ## each element of observed_data is a list with A (the thinned
    ## points) lambda (the gp at the unthinned points), and x (the
    ## unthinned points).
    A = complete_data['A']
    A_tilde = complete_data['A_tilde']
    g = complete_data['g']
    
    N = np.size(g)
    K = make_se_kernel(np.append(np.array(A), np.array(A_tilde)), params['lengthscale'], sampling_sd, params['gp_ridge'])
    lambda_of_x = np.random.multivariate_normal(mean = np.zeros(N), cov = K)
    g_sample = g + lambda_of_x
    
    is_log_prob = sum(np.log(scipy.stats.norm.pdf(g_sample - g, loc = 0, scale = sampling_sd)))
    complete_data_log_prob,pois,thinning, gp = sgcp_complete_data_log_prob(A, A_tilde, g_sample, params['base_rate'], params['lengthscale'], params['gp_sigma'], params['gp_ridge'])
    log_w = complete_data_log_prob - is_log_prob
    return ({ "A" : A, "A_tilde" : A_tilde, "g" : g_sample, "w" : np.exp(log_w)},pois,thinning, gp)
def sequence_complete_data_log_prob(seq, gl_seq, A, A_tilde, g, model_params):
    lambda_star = model_params['lambda_star']
    lengthscale = model_params['lengthscale']
    sigma = model_params['sigma']
    gp_ridge = model_params['gp_ridge']
    ber_params = model_params['ber_params']
    sgcp_log_prob, pois,thinning, gp = sgcp_complete_data_log_prob(A, A_tilde, g, lambda_star, lengthscale, sigma, gp_ridge)
    sequence_given_sgcp_log_prob = sequence_log_prob_given_lesions(seq, gl_seq, A, ber_params)
    return(sgcp_log_prob + sequence_given_sgcp_log_prob)
def sgcp_complete_data_log_prob(A, A_tilde, g, lambda_star, lengthscale, sigma, gp_ridge):
    K = len(A)
    M = len(A_tilde)
    pois = (K + M) * np.log(lambda_star) - lambda_star - math.lgamma(K + M + 1)
    thinning = thinning_log_prob(g, K, M)
    gp_kern_ridged = make_se_kernel(np.append(np.array(A), np.array(A_tilde)), lengthscale, sigma, gp_ridge)
    gp = np.log(scipy.stats.multivariate_normal.pdf(g, mean = np.zeros(len(g)), cov = gp_kern_ridged))
    
    return(pois + thinning + gp, pois, thinning, gp)
def thinning_log_prob(g, K, M):
    if(K == 0 and M == 0):
        return(0)
    elif(K == 0 and M > 0):
        return(sum(np.log(logistic(-g))))
    elif(K > 0 and M == 0):
        return(sum(np.log(logistic(g))))
    else:
        return(sum(np.log(logistic(g[range(K)]))) + sum(np.log(logistic(-g[range(K, K+M)]))))
def logistic(x):
    return(1 / (1 + np.exp(-x)))

def logit(p):
    return(np.log(p / (1 - p)))
def discrete_to_interval(i, seq_length):
    return (i - np.random.uniform(low = 0, high = 1, size = 1)) / seq_length
def lengthscale_inference(x_list, g_list, weights, l_test_grid, model_params, full_grid = False):
    log_probs = np.zeros(len(l_test_grid))
    for (ls, ls_idx) in zip(l_test_grid, range(len(l_test_grid))):
        g_log_p_at_test = 0.
        for i in range(len(x_list)):
            g = g_list[i]
            K = make_se_kernel(x_list[i], lengthscale = ls, sigma = model_params['gp_sigma'], gp_ridge = model_params['gp_ridge'])
            g_log_p_at_test += weights[i] * np.log(scipy.stats.multivariate_normal.pdf(g, cov = K))
        log_probs[ls_idx] = g_log_p_at_test
    if full_grid:
        return(l_test_grid,log_probs)
    else:
        return(l_test_grid[np.argmax(log_probs)])