# Load our stuff
import keras
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
import pandas as pd
import glob
from Bio.Alphabet.IUPAC import unambiguous_dna, ambiguous_dna
from random import sample
from sim_data.sumstats import *
# Set data path (CLI?)
data_path = 'sim_data/'
params = {'lengthscale':-10.95518256,
        'gp_sigma':5.62844507,
        'p_fw':0.56139752,
        'exo_left':  15.24284578,
        'exo_right':18.89962825,
        'ber_prob': 0.51447639,
        'fw_br' : 0.5,
        'rc_br': 0.5,
        'gp_offset': -10.0,
        "gp_ridge" : .04}
# Load df with all seqs
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
seqs = []
for i in range(50):
            t_seq = gen_batch_letters(Seq(parent_sequences[i], unambiguous_dna),1, params)[0]
            seqs.append(t_seq)


data = pd.DataFrame(np.concatenate((seqs,parent_sequences)))
data.to_csv('data.csv', mode='a', index=False, header=False)