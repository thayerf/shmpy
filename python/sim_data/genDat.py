# Load our stuff
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
import csv
import collections
from scipy.stats import norm
# Load options
import pandas as pd
import glob
from Bio.Alphabet.IUPAC import unambiguous_dna, ambiguous_dna
from random import sample
from sumstats import *
from params import *
# Load df with all seqs
df = pd.read_pickle("./data/full_edge_df.pk1")
parent_sequences = df['orig_seq']
run = sys.argv[0]
cm = ContextModel(3, 2, pkgutil.get_data("SHMModels", "data/aid_goodman.csv"))

# Get batch (BER AND POL ETA DEFINED HERE)
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
    exo_left = 1.0/prior_params['exo_left']
    exo_right = 1.0/prior_params['exo_right']
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
# Generate batch of Param, Seq batch pairs
def gen_batch(batch_size, num_seqs, parent_sequences):
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
        obs_sample = []
        parent_sample = sample(list(parent_sequences),num_seqs)
        for i in range(num_seqs):
            t_seq = gen_batch_letters(Seq(parent_sample[i], unambiguous_dna),1, true_model_params)
            obs_sample.append(t_seq[0])
        colocals = colocal_vector(parent_sample, obs_sample)
        exo = np.concatenate([get_exo_summ(obs_sample[i],parent_sample[i]) for i in range(len(parent_sample))]).ravel().tolist()
        at = np.concatenate([get_pairwise_at(obs_sample[i],parent_sample[i]) for i in range(len(parent_sample))]).ravel().tolist()
        c_counts, g_counts = get_cg_summ_sample(obs_sample,parent_sample)
        bp = base_prob(obs_sample,parent_sample)
        atp = at_frac(obs_sample,parent_sample)
        summ_stat = np.append(colocals,[np.mean(exo),np.mean(at), np.sum(c_counts)/(np.sum(c_counts) + np.sum(g_counts)), bp,atp])
        X.append(summ_stat)
        theta.append(true_params_array)
    return X,theta

test_batch_size = 10
train_batch_size = 100
num_seqs = 2000

test_X, test_theta = gen_batch(test_batch_size, num_seqs,parent_sequences)
train_X, train_theta = gen_batch(train_batch_size, num_seqs,parent_sequences)

if np.isnan(test_X).any() or np.isnan(train_X).any():
    pass
else:
    data = pd.DataFrame(np.concatenate((test_X,test_theta,np.ones((test_batch_size,1))),axis = 1))
    data = data.append(pd.DataFrame(np.concatenate((train_X,train_theta,np.zeros((train_batch_size,1))),axis = 1)))
    data.to_csv('data.csv', mode='a', index=False, header=False)