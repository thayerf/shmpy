# Load our stuff
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from Bio import SeqIO
from SHMModels.simulate_mutations import *
from SHMModels.fitted_models import ContextModel
import pkgutil
import logging
import os
import sys
import json
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
random.seed(1408)
import csv
import collections
# Load options

germline_sequence = "data/gpt.fasta"
germline =list(SeqIO.parse(germline_sequence, "fasta"))[0].seq

cm = ContextModel(3, 2, pkgutil.get_data("SHMModels", "data/aid_goodman.csv"))

pol_eta_params = {
            "A": [0.9, 0.02, 0.02, 0.06],
            "G": [0.01, 0.97, 0.01, 0.01],
            "C": [0.01, 0.01, 0.97, 0.01],
            "T": [0.06, 0.02, 0.02, 0.9],
        }
ber_params = np.array([0.25,0.25,0.25,0.25])


fw_contexts = np.zeros(308)
for i in range(308):
    fw_contexts[i] = cm.get_context_prob(i,germline)
rc_contexts = np.zeros(308)
for i in range(308):
    rc_contexts[i] = cm.get_context_prob(i,germline.complement())
    
def sample_prior():
    ls = np.random.uniform(low = -12.0, high = -2.0)
    sg = np.random.uniform(low = 5.0, high = 15.0)
    off = -10
    p_fw = np.random.uniform(low =0.0, high = 1.0)
    exo_left = np.random.uniform(low =0.5, high = 0.5)
    exo_right = np.random.uniform(low =0.5, high = 0.5)
    ber_prob = np.random.uniform(low = 0.0, high = 1.0)
    thinning_prob = norm.cdf(10.0/sg)
    fw_br = np.minimum(fw_contexts/(1.0-thinning_prob),1.0)
    rc_br = np.minimum(rc_contexts/(1.0-thinning_prob),1.0)
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

# Get batch
def gen_batch_letters(seq,batch_size, params):
       # The prior specification
    ber_prob = params['ber_prob']
    ber_params = [0.25,0.25,0.25,0.25]
    
    bubble_size = 25.0
    pol_eta_params = {
        "A": [0.9, 0.02, 0.02, 0.06],
        "G": [0.01, 0.97, 0.01, 0.01],
        "C": [0.01, 0.01, 0.97, 0.01],
        "T": [0.06, 0.02, 0.02, 0.9],
    }
    prior_params = params
    exo_left = prior_params['exo_left']
    exo_right = prior_params['exo_right']
    mutated_seq_list = []
    for i in range(batch_size):
          mr = MutationRound(
          seq,
          ber_lambda=1.0,
          mmr_lambda=(1 - ber_prob)/ber_prob,
          replication_time=100,
          bubble_size=bubble_size,
          aid_time=10,
          exo_params={"left": exo_left, "right": exo_right},
          pol_eta_params=pol_eta_params,
          ber_params=ber_params,
          p_fw= prior_params['p_fw'],
          aid_context_model=cm,
          log_ls = prior_params['lengthscale'],
          sg = prior_params['gp_sigma'],
          fw_br = prior_params['fw_br'],
          rc_br = prior_params['rc_br'],
          off = prior_params['gp_offset']
          )
          mr.mutation_round()
          mutated_seq_list.append(SeqRecord(mr.repaired_sequence, id=""))
    return [list(i.seq) for i in mutated_seq_list]

def site_dist_colocal(seqs,germline,base_probs,dist):
    values = np.zeros(len(base_probs))
    vars = np.zeros(len(base_probs))
    for i in range(len(base_probs)-dist):
        if base_probs[i]*base_probs[i+dist]>0:
            p_1 = np.mean([z[i]!= germline[i] and z[i+dist]!=germline[i+dist] for z in seqs])
            p_2 = base_probs[i]
            p_3 = base_probs[i+dist]
            values[i] = p_1/(p_2*p_3)
            vars[i] = (1-p_2-p_3+p_2*p_3)/(len(seqs)*p_2*p_3)
        else:
            values[i] = 0.00
            vars[i] = 0.0
    return(values,vars)
def get_colocal(seqs,germline,base_probs, max_dist):
    colocals = np.zeros(max_dist)
    for i in range(max_dist):
        values,vars = site_dist_colocal(seqs,germline, base_probs, i+1)
        colocals[i] = np.nansum(values[values>0.0]/np.sqrt(vars[values>0.0]))/np.sum(1/np.sqrt(vars[vars>0.0]))
    return(colocals)

def gauss_kernel(x,y,eps):
    return np.exp(-(np.sum(np.square(x-y)))/(2*eps**2))

def shortestDistance(S, X):
 
    # Find distance from occurrences of X
    # appearing before current character.
    inf = float('inf')
    prev = inf
    ans = []
    for i,j in enumerate(S):
        if S[i] == X:
            prev = i
        if (prev == inf) :
            ans.append(inf)
        else :    
            ans.append(i - prev)
 
 
    # Find distance from occurrences of X
    # appearing after current character and
    # compare this distance with earlier.
    prev = inf
    for i in range(len(S) - 1, -1, -1):
        if S[i] == X:
            prev = i
        if (X != inf):   
            ans[i] = min(ans[i], prev - i)
 
    # return array of distance
    return ans

def get_mmr_summ(seqs, germline):
    mut_ind = [i != np.array(list(germline)) for i in seqs]
    dists = np.minimum(shortestDistance(germline,'C'),shortestDistance(germline,'G'))
    avg_d_num = 0
    avg_d_deno = 0
    for i in range(len(seqs)):
        avg_d_deno += np.sum(mut_ind[i])
        avg_d_num += np.sum(dists[mut_ind[i]])
    c_mut_count = 0
    g_mut_count = 0
    for i in seqs:
        c_mut_count += np.sum(np.logical_and(i != np.array(list(germline)),np.array([j == 'C' for j in germline])))
        g_mut_count += np.sum(np.logical_and(i != np.array(list(germline)),np.array([j == 'G' for j in germline])))
    return c_mut_count/(c_mut_count+g_mut_count), avg_d_num/avg_d_deno

def importance_sample(obs_sequences,germline,n_imp_samp, n, eps):
    
    true_bp = (1.0-np.mean(obs_sequences == np.array(list(germline)), axis = 0))
    mmr_stat = get_mmr_summ(obs_sequences, germline)
    colocals = get_colocal(obs_sequences, germline,true_bp, 50)
    base_colocal = np.append(colocals[0:50:5], np.mean(true_bp))
    base_colocal = np.append(base_colocal, mmr_stat)
    ls_list = []
    w_list = []
    sg_list = []
    rate_list = []
    p_fw_list = []
    ber_p_list = []
    for i in range(n_imp_samp):
        
        model_params = sample_prior()
        sample = gen_batch_letters(germline, n, model_params)
        
        sample_bp = (1.0-np.mean(sample == np.array(list(germline)), axis = 0))
        
        sample_colocals = get_colocal(sample,germline,sample_bp,50)
        sample_mmr_stat = get_mmr_summ(sample, germline)
        colocal = np.append(sample_colocals[0:50:5], np.mean(sample_bp))
        colocal = np.append(colocal, sample_mmr_stat)
        w = gauss_kernel(colocal,base_colocal,eps)
        if math.isnan(w):
            w = 0.0
        w_list.append(w)
        ls_list.append(model_params['lengthscale'])
        sg_list.append(model_params['gp_sigma'])
        rate_list.append(model_params['base_rate'])
        p_fw_list.append(model_params['p_fw'])
        ber_p_list.append(model_params['ber_prob'])
        if i % 50 == 0:
            print(i)
    return rate_list, ls_list, sg_list, p_fw_list, ber_p_list,  w_list, base_colocal

def gen_batch(batch_size, n_seqs):
    X = []
    theta = []
    for i in range(batch_size):
        true_model_params = sample_prior()
        true_params_array = np.array((true_model_params['lengthscale'],
                                     true_model_params['gp_sigma'],
                                     true_model_params['p_fw'],
                                     true_model_params['exo_left'],
                                     true_model_params['exo_right'],
                                     true_model_params['ber_prob']))
        obs_sample = gen_batch_letters(germline, n_seqs, true_model_params)
        true_bp = (1.0-np.mean(obs_sample == np.array(list(germline)), axis = 0))
        mmr_stat = get_mmr_summ(obs_sample, germline)
        colocals = np.convolve(get_colocal(obs_sample, germline,true_bp, 50),np.ones(5),'valid')/5.0
        summ_stat = np.concatenate((colocals,mmr_stat))
        summ_stat = np.append(summ_stat, np.array(np.mean(true_bp)))
        X.append(summ_stat)
        theta.append(true_params_array)
    return X,theta

def genTraining(batch_size, train_n):
    while True:
        # Get training data for step
        batch_data,batch_labels = gen_batch(batch_size, train_n)
        yield (
            np.array(batch_data),
            [
                np.array(batch_labels)
            ],
        )

model = Sequential()
model.add(Dense(49, input_dim= 49, activation='relu'))
model.add(Dense(49, activation='relu'))
model.add(Dense(49, activation='relu'))
model.add(Dense(6, activation='linear'))



hist = keras.callbacks.History()
# Give summary of architecture

# Initialize optimizer with given step size
adam = Adam(lr=.02)
# Compile model w/ pinball loss, use mse as metric
model.compile(
    loss=['mse'],
    optimizer=adam,
)

print(model.summary(90))
# Train the model using generator and callbacks we defined w/ test set as validation

history = model.fit_generator(
    genTraining(50, 1000),
    epochs=200,
    steps_per_epoch=1,
    callbacks=[hist],
)

test_data, test_labels = gen_batch(100,1000)

pred_labels = model.predict(np.array(test_data))
np.save('labels',np.array(test_labels))
np.save('preds', np.array(pred_labels))
