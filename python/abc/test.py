# Load our stuff
import pkgutil
from pathlib import Path
import numpy as np
from Bio import SeqIO
from SHMModels.fitted_models import ContextModel
from keras import optimizers, Input, Model
from keras.models import Sequential
import keras.backend as K
import matplotlib.pyplot as plt
from keras.layers import (
    Dense,
    TimeDistributed,
    SimpleRNN,
    Dropout,
    Conv2D,
    Conv2DTranspose,
    Flatten,
    Conv1D,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
)
import scipy
import math

##### USER INPUTS (Edit some of these to be CLI eventually)

# Path to germline sequence
germline_sequence = "../data/gpt.fasta"
# Context model length and pos_mutating
context_model_length = 3
context_model_pos_mutating = 2
# Path to aid model
aid_context_model = "data/aid_logistic_3mer.csv"
# Num seqs and n_mutation rounds
n_seqs = 1
n_mutation_rounds = 1
# step size
step_size = 0.01
# batch size num epochs
batch_size = 500
gl1 = list(list(
    SeqIO.parse(germline_sequence, "fasta")
)[0].seq)
germline = list(list(
    SeqIO.parse(germline_sequence, "fasta")
)[0].seq)

# First, we want to identify our true parameter set and our starting parameter set.
# GP OFFSET FIXED AT 12
n_samples = 10;max_steps = 4;n_imp_samples = 1;sampling_noise_sd = .01
def sample_prior():
    ls = np.random.uniform(low = 0, high = 0.075)
    return { "base_rate" : 2.0,
                       "lengthscale" :ls,
                       "gp_sigma" : 12.0,
                       "gp_ridge" : .01,
            "gp_offset": -12
            }
    
n = np.size(germline)
c_array = np.zeros(n)
for i in range(n):
    c_array[i] = 1.0 * (germline[i] == 'C')

def make_se_kernel(x, lengthscale, sigma, gp_ridge):
    D = np.zeros([len(x), len(x)])
    upper_tri = np.triu_indices(len(x), 1)
    D[upper_tri] = ((np.array(x)[upper_tri[0]] - np.array(x)[upper_tri[1]])**2)
    D += D.T
    K = sigma**2 * np.exp(-D / (2 * lengthscale))
    np.fill_diagonal(K, K.diagonal() + gp_ridge)
    return K

def forward_sample_sequence_sgcp(sequence, params):
    n = np.size(sequence)
    vals = np.random.poisson(lam = params['base_rate'], size = n)
    # Only include prelesions at c site
    vals = np.multiply(vals, c_array)
    x = []
    for i in range(np.size(vals)):
        if vals[i] > 0:
            conts = (i-np.random.uniform(size = 1))/n
            x = np.append(x, conts)
    N = np.size(x)
    if N > 0:
        K = make_se_kernel(x, params['lengthscale'], params['gp_sigma'], params['gp_ridge'])
        lambda_of_x = np.random.multivariate_normal(mean = np.zeros(N)+params['gp_offset'], cov = K)
        sigma_lambda_of_x = 1 / (1 + np.exp(lambda_of_x))
        uniforms = np.random.uniform(low = 0, high = 1, size = len(x))
        A_and_g = [(xi, li) for (xi, si, li, u) in zip(x, sigma_lambda_of_x, lambda_of_x, uniforms) if 1 < li]
        A_tilde_and_g = [(xi, li) for (xi, si, li, u) in zip(x, sigma_lambda_of_x, lambda_of_x, uniforms) if 1 >= li]
        A = [a for (a, g) in A_and_g]
        A_tilde = [at for (at, g) in A_tilde_and_g]
        g = [g for (a, g) in A_and_g + A_tilde_and_g]
    else:
        A = []
        A_tilde = []
        g = []
    return({ "A" : A, "A_tilde" : A_tilde, "g" : g})

def sample_seq(sequence, params):
    new_seq = sequence[:]
    res = forward_sample_sequence_sgcp(sequence,params)
    A = np.unique(np.ceil(np.multiply(res['A'],len(sequence))))
    muts = np.random.choice(a = ['A','G','T'], size = len(A))
    for i in range(len(A)):
        new_seq[int(A[i])] = muts[i]
    A_tilde = np.unique(np.ceil(np.multiply(res['A_tilde'],len(sequence))))
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
def weight_array(ar, weights):
     sort = weights[np.argsort(ar)].cumsum()/np.sum(weights)
     sort_ar = np.sort(ar)
     return sort_ar,sort

# Get a 2d hot encoding of a sequence
def hot_encode_2d(seq):
    seq_hot = np.zeros((len(seq), 4, 1))
    for j in range(len(seq)):
        seq_hot[j, 0, 0] = seq[j] == "A"
        seq_hot[j, 1, 0] = seq[j] == "T"
        seq_hot[j, 2, 0] = seq[j] == "G"
        seq_hot[j, 3, 0] = seq[j] == "C"
    return seq_hot

def gen_batch(seq, params, batch_size):
    mut = []
    les = []
    preles = []
    gaus = []
    for i in range(batch_size):
        temp = sample_seq(seq, params)
        mut.append(temp['seq'])
        les.append(temp['A'])
        preles.append(temp['A_tilde'])
        gaus.append(temp['g'])
    return np.array(mut), np.array(les), np.array(preles), np.array(gaus)

def get_colocal(batch_size, model_params, sample = None):
    if sample is None:
        sample,A, A_t, g = gen_batch(germline, model_params, batch_size)
    base_prob = (1.0-np.mean(sample == germline))*(308.0/70.0)
    num = np.zeros(30)
    deno = np.zeros(30)
    for k in range(batch_size):
        for i in range(308):
            if c_array[i]:
                for j in range(i+1,np.min([i+30,307])):
                    if c_array[j]:
                        if sample[k,i] != germline[i]:
                            num[j-i] = num[j-i] + np.float(sample[k,j] != germline[j])
                        deno[j-i] = deno[j-i] + 1.0
    return num[1:],deno[1:],base_prob, sample
def gauss_kernel(x,y,eps):
    return np.exp(-(np.sum(np.square(x-y)))/(2*eps**2))

def importance_sample(obs_sequences,n_imp_samp, n, eps):
    num, deno, base_prob, sample = get_colocal(n,true_model_params, sample = obs_sequences)
    base_colocal = num/(deno*base_prob**2)
    ls_list = []
    w_list = []
    for i in range(n_imp_samp):
        model_params = sample_prior()
        num, deno, base_prob, sample = get_colocal(n,model_params)
        colocal = num/(deno*base_prob**2)
        w_list.append(gauss_kernel(colocal[0:30:3], base_colocal[0:30:3],eps))
        ls_list.append(model_params['lengthscale'])
        if i % 50 == 0:
            print(i)
    return ls_list, w_list, base_colocal

true_model_params = sample_prior()
obs_sample,A, A_t, g = gen_batch(germline,true_model_params, 300)

ls_list, w_list, base_colocal = importance_sample(obs_sample, 1000, 300, 0.2)



pred_mean = np.dot(w_list,ls_list)/np.sum(w_list)
true = true_model_params['lengthscale']

f = open("est", "a")
f.write(str(pred_mean) + " ")
f.close()

f = open("true", "a")
f.write(str(true) + " ")
f.close()
