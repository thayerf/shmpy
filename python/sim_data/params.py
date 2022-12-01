import numpy as np
from scipy.stats import norm
# This file defines all of the unchanged parameters and hyperparameters which govern forward simulation.
def sample_prior():
    # Sample from the prior for all parameters
    # Lengthscale
    ls = np.random.uniform(low = -12.0, high = -2.0)
    # Variance of the Gaussian Process
    sg = np.random.uniform(low = 5.0, high = 15.0)
    # Gaussian Process mean (unchanged)
    off = -10
    # Probability of the forward strand
    p_fw = np.random.uniform(low =0.0, high = 1.0)
    # Geometric prob for size of exo stripping region to left and right
    exo_left = np.random.uniform(low =0.1, high = 0.7)
    exo_right = np.random.uniform(low =0.1, high = 0.7)
    # Probability that BER is recruited
    ber_prob = np.random.uniform(low = 0.0, high = 1.0)
    # Probability of thinning a prelesion
    thinning_prob = norm.cdf(10.0/sg)
    # Base rate on each strand
    fw_br = 0.5
    rc_br = 0.5
    return {           "lengthscale" : ls,
                       "gp_sigma" : sg,
                       "gp_ridge" : .04,
            "gp_offset": off,
            "p_fw": p_fw,
            "fw_br": fw_br,
            "rc_br": rc_br,
            "exo_left": exo_left,
            "exo_right": exo_right,
            "ber_prob": ber_prob
            }