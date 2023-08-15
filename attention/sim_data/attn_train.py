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
from scipy.stats import norm
import csv
import collections
from scipy.stats import norm
# Load options
import pandas as pd
import glob
from random import sample
from sumstats import *
from params import *
import math
import os
import torch
import torch.nn as nn
import keras
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Load df with all seqs
df = pd.read_pickle("./data/full_edge_df.pk1")
parent_sequences = df['orig_seq']
load = sys.argv[0]
cm = ContextModel(3, 2, pkgutil.get_data("SHMModels", "data/aid_goodman.csv"))

train_batch_size = 10
train_data_size = 500
num_seqs = 2000


def hot_encode_dna_sequences(sequences):
    bases = ['A', 'C', 'G', 'T']
    num_bases = len(bases)
    
    # Create a dictionary to map bases to indices
    base_to_index = {base: index for index, base in enumerate(bases)}
    
    # Get the maximum length of the sequences
    max_length = max(len(seq) for seq in sequences)
    
    # Initialize an empty matrix to store the hot encoded sequences
    encoded_sequences = np.zeros((len(sequences), max_length, num_bases), dtype=int)
    
    # Iterate over each sequence and hot encode it
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            if base in base_to_index:
                encoded_sequences[i, j, base_to_index[base]] = 1
            else:
                # Handle unrecognized bases as desired (e.g., ignore or treat as zeros)
                pass
    
    return encoded_sequences

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
    X_par = []
    X_chi = []
    theta = []
    summ_stats = []
    for i in range(batch_size):
        true_model_params = sample_prior()
        true_params_array = np.array((true_model_params['lengthscale'],
                                     true_model_params['gp_sigma'],
                                     true_model_params['p_fw'],
                                     true_model_params['exo_left'] + true_model_params['exo_right'],
                                     true_model_params['ber_prob']))
        obs_sample = []
        parent_sample = sample(list(parent_sequences),num_seqs)
        for i in range(num_seqs):
            t_seq = gen_batch_letters(Seq(parent_sample[i]),1, true_model_params)
            obs_sample.append(t_seq[0])
        colocals = colocal_vector(parent_sample, obs_sample)
        exo = np.concatenate([get_exo_summ(obs_sample[i],parent_sample[i]) for i in range(len(parent_sample))]).ravel().tolist()
        at = np.concatenate([get_pairwise_at(obs_sample[i],parent_sample[i]) for i in range(len(parent_sample))]).ravel().tolist()
        c_counts, g_counts = get_cg_summ_sample(obs_sample,parent_sample)
        bp = base_prob(obs_sample,parent_sample)
        atp = at_frac(obs_sample,parent_sample)
        summ_stat = np.append(colocals,[np.mean(exo),np.mean(at), np.sum(c_counts)/(np.sum(c_counts) + np.sum(g_counts)), bp,atp])
        summ_stats.append(summ_stat)
        X_chi.append(hot_encode_dna_sequences(obs_sample))
        X_par.append(hot_encode_dna_sequences(parent_sample))
        theta.append(true_params_array)
    return X_chi,X_par,theta, summ_stats


train_X_chi,train_X_par, train_theta, summ_stats = gen_batch(train_data_size, num_seqs,parent_sequences)
summ_stats = np.nan_to_num(np.array(summ_stats))
summ_stats = (summ_stats - np.mean(summ_stats,axis = 0))/np.std(summ_stats, axis = 0)
training = [train_X_chi[i]-train_X_par[i] for i in range(len(train_X_chi))]
length = np.max([np.shape(i)[1] for i in train_X_chi])

for i in range(len(training)):
    if np.shape(training[i])[1] < length:
        dist = length - np.shape(training[i])[1]
        training[i] = np.concatenate((training[i],np.zeros((num_seqs,dist,4))), axis = 1)



class AttentionModel(nn.Module):
    def __init__(self, seq_length, conv_filters,
                 dropout):
        super().__init__()
        self.seq_length = seq_length
        self.conv_filters = conv_filters
        self.device = device

        self.dropout = nn.Dropout(p=dropout)

        self.conv = nn.Conv2d(1,self.conv_filters,(10,4), padding = 'same')
        self.enclayer = nn.TransformerEncoderLayer(d_model= 4*self.conv_filters, nhead = 1, dim_feedforward= 128,batch_first=True)
        self.linear = nn.Linear(self.seq_length*self.conv_filters*4, out_features= 256)
    def forward(self, x):

        x = x.to(self.device) 

        conv_out = self.conv(x)
        conv_out = torch.reshape(conv_out,(-1,self.seq_length,4*self.conv_filters))
        attn_out = self.enclayer(conv_out)
        attn_out = torch.reshape(attn_out, (-1,self.seq_length*self.conv_filters*4))
        out = self.linear(attn_out)
        return out
if load == 1:
    model = AttentionModel(length,10, 0.03)
    model.load_state_dict(torch.load('preds/model_hybrid_exo_split2'))
    model.cuda()
    final_linear = torch.load('preds/linear_hybrid_exo_split2')
else:
    model = AttentionModel(length,10, 0.03)
    final_linear = nn.Linear(262,5)
    model = model.to(device)
    final_linear = final_linear.to(device)
summ_model = keras.models.load_model('model')
param_means = np.array([-7.0,10.0,0.5,10.5,10.5,0.5])
param_sds = (1/np.sqrt(12))*np.array([10,10,1,18,18,1])

summ_preds = summ_model.predict(summ_stats)

random.seed(1408)
loss = nn.MSELoss()
optimizer = torch.optim.Adam([{'params':model.parameters()}, {'params':final_linear.parameters()}], lr=1e-6)

losses = []
for iter in range(20000):
    idx = random.sample(range(0, train_data_size), train_batch_size)
    batch_training = [training[i] for i in idx]
    training_tensors = [torch.reshape(torch.tensor(i).float(), (num_seqs,1,length,4)).cuda() for i in batch_training]
    training_labels = torch.tensor(((train_theta-np.mean(train_theta,axis = 0))/np.std(train_theta,axis = 0))[idx]).float().cuda()
    training_summ = torch.tensor(summ_preds[idx,:]).float().cuda()
    optimizer.zero_grad()
    embeds = [torch.mean(model(i),0) for i in training_tensors]
    embeds = torch.cat((torch.stack(embeds),training_summ),dim = 1)
    preds = final_linear(embeds)
    output = loss(preds, training_labels)
    losses.append(output.item())
    if(iter % 40 == 0):
        print(iter)
        print(np.mean(losses[-30:]))
    if(iter % 100 == 0):
        torch.save(model.state_dict(), 'preds/model_hybrid_exo_split2')
        torch.save(final_linear,'preds/linear_hybrid_exo_split2')
    output.backward()
    optimizer.step()
    del training_tensors
    del training_labels
    del training_summ
    del embeds
    del preds
    