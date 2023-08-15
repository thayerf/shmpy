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
import sys
# Load df with all seqs
true_model_params = {           "lengthscale" : -9.71396504,
                       "gp_sigma" : 6.9732454,
                       "gp_ridge" : .04,
            "gp_offset": -10,
            "p_fw": 0.51655939,
            "fw_br": 0.5,
            "rc_br": 0.5,
            "exo_left": 18.22189314,
            "exo_right": 14.75165542,
            "ber_prob": 0.657664
            }
df = pd.read_pickle("full_edge_df.pk1")
df = df[0:50]
parent_sequences = df['PARENT_SEQ']
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


num_seqs = 50

obs_sample = []
orig_sample = []
parent_sample = parent_sequences
for i in range(num_seqs):
    t_seq = gen_batch_letters(Seq(parent_sample[i], unambiguous_dna),1, true_model_params)
    t_str = "".join([str(i) for i in t_seq[0]])
    obs_sample.append(t_str)
    orig_sample.append(parent_sample[i])

data = pd.DataFrame(list(zip(obs_sample,orig_sample)), columns=['mut','orig'])
data.to_csv('./data/' + str(sys.argv[1]) + '.csv', mode='a', index=False, header=False)