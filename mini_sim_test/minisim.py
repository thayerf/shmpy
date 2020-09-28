# Load our stuff

import numpy as np

def make_se_kernel(x, lengthscale, sigma, gp_ridge):
    D = np.zeros([len(x), len(x)])
    upper_tri = np.triu_indices(len(x), 1)
    D[upper_tri] = ((np.array(x)[upper_tri[0]] - np.array(x)[upper_tri[1]])**2)
    D += D.T
    K = sigma**2 * np.exp(-D / (2 * lengthscale))
    np.fill_diagonal(K, K.diagonal() + gp_ridge)
    return K

def forward_sample_sequence_sgcp(sequence, params,c_array):
    n = np.size(sequence)
    vals = np.random.poisson(lam = params['base_rate'], size = n)
    # Only include prelesions at c site
    vals = np.multiply(vals, c_array)
    x = []
    for i in range(np.size(vals)):
        if vals[i] > 0:
            conts = (i-np.random.uniform(size = int(vals[i])))/n
            x = np.append(x, conts)
    N = np.size(x)
    K = make_se_kernel(x, params['lengthscale'], params['gp_sigma'], params['gp_ridge'])
    lambda_of_x = np.random.multivariate_normal(mean = np.zeros(N), cov = K)
    sigma_lambda_of_x = 1 / (1 + np.exp(-lambda_of_x))
    uniforms = np.random.uniform(low = 0, high = 1, size = len(x))
    A_and_g = [(xi, li) for (xi, si, li, u) in zip(x, sigma_lambda_of_x, lambda_of_x, uniforms) if u < si]
    A_tilde_and_g = [(xi, li) for (xi, si, li, u) in zip(x, sigma_lambda_of_x, lambda_of_x, uniforms) if u >= si]
    A = [a for (a, g) in A_and_g]
    A_tilde = [at for (at, g) in A_tilde_and_g]
    g = [g for (a, g) in A_and_g + A_tilde_and_g]
    return({ "A" : A, "A_tilde" : A_tilde, "g" : g})

def sample_seq(sequence, params,c_array):
    new_seq = sequence[:]
    res = forward_sample_sequence_sgcp(sequence,params,c_array)
    A = np.unique(np.ceil(np.multiply(res['A'],len(sequence))))
    muts = np.random.choice(a = ['A','G','C','T'], size = len(A))
    for i in range(len(A)):
        new_seq[int(A[i])] = muts[i]
    A = (np.ceil(np.multiply(res['A'],len(sequence))))
    A_tilde = (np.ceil(np.multiply(res['A_tilde'],len(sequence))))
    g = res['g']
    return({"seq": new_seq,  "A" : A, "A_tilde" : A_tilde, "g" : g})
def process_latent(seq_res):
    n = len(seq_res['seq'])
    A = seq_res['A']
    A_tilde = seq_res['A_tilde']
    g = seq_res['g']
    A_long = np.zeros(n)
    A_tilde_long = np.zeros(n)
    g_long = np.zeros(n)
    k = 0
    for i in A:
        A_long[int(i)] = A_long[int(i)]+1
        g_long[int(i)] = g[k]
        k = k+1
    for i in A_tilde:
        A_tilde_long[int(i)] = A_tilde_long[int(i)]+1
        g_long[int(i)] = g[k]
        k = k+1
    return(np.stack([g_long,A_long, A_tilde_long],axis = 1))
    
# Get a 2d hot encoding of a sequence
def hot_encode_2d(seq):
    seq_hot = np.zeros((len(seq), 4, 1))
    for j in range(len(seq)):
        seq_hot[j, 0, 0] = seq[j] == "A"
        seq_hot[j, 1, 0] = seq[j] == "T"
        seq_hot[j, 2, 0] = seq[j] == "G"
        seq_hot[j, 3, 0] = seq[j] == "C"
    return seq_hot

def gen_batch(seq, params, batch_size,c_array):
    mut = []
    les = []
    for i in range(batch_size):
        temp = sample_seq(seq, params,c_array)
        mut.append(hot_encode_2d(temp['seq']))
        les.append(process_latent(temp))
    return np.array(mut),np.array(les)