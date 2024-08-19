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
#from Bio.Alphabet.IUPAC import unambiguous_dna, ambiguous_dna
from random import sample
from sumstats import *
from params import *
# Load df with all seqs
df = pd.read_pickle("./data/full_edge_df.pk1")
parent_sequences = df['orig_seq']
run = sys.argv[0]
cm = ContextModel(3, 2, pkgutil.get_data("SHMModels", "data/aid_goodman.csv"))

def hot_encode_dna_sequences(sequences):
    bases = ['A', 'C', 'G', 'T']
    num_bases = len(bases)
    
    # Create a dictionary to map bases to indices
    base_to_index = {base: index for index, base in enumerate(bases)}
    
    # Get the maximum length of the sequences
    max_length = max(len(seq) for seq in sequences)
    
    # Initialize an empty matrix to store the hot encoded sequences
    encoded_sequences = np.zeros((len(sequences), max_length, num_bases), dtype=np.int)
    
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
            t_seq = gen_batch_letters(Seq(parent_sample[i]),1, true_model_params)
            obs_sample.append(t_seq[0])
        X_chi.append(hot_encode_dna_sequences(obs_sample))
        X_par.append(hot_encode_dna_sequences(parent_sample))
        theta.append(true_params_array)
    return X_chi,X_par,theta


train_batch_size = 100
num_seqs = 500

train_X_chi,train_X_par, train_theta = gen_batch(train_batch_size, num_seqs,parent_sequences)

training = [train_X_chi[i]-train_X_par[i] for i in range(len(train_X_chi))]
length = np.max([np.shape(i)[1] for i in train_X_chi])

for i in range(len(training)):
    if np.shape(training[i])[1] < length:
        dist = length - np.shape(training[i])[1]
        training[i] = np.concatenate((training[i],np.zeros((num_seqs,dist,4))), axis = 1)

import math
import os
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.nn.functional as F
class AttentionModel(nn.Module):
    def __init__(self, seq_length, conv_filters,
                 dropout):
        super().__init__()
        self.seq_length = seq_length
        self.conv_filters = conv_filters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

model = AttentionModel(424,10, 0.0)
final_linear = nn.Linear(256,6)

random.seed(1408)
training_tensors = [torch.reshape(torch.tensor(i).float(), (500,1,424,4)) for i in training]
training_labels = torch.tensor((train_theta-np.mean(train_theta,axis = 0))/np.std(train_theta,axis = 0)).float()

loss = nn.MSELoss()
optimizer = torch.optim.Adam([{'params':model.parameters()}, {'params':final_linear.parameters()}], lr=1e-5)

losses = []
for iter in range(1000):
    optimizer.zero_grad()
    embeds = [torch.mean(model(i),0) for i in training_tensors]
    preds = final_linear(torch.stack(embeds))
    output = loss(preds, training_labels)
    print(output.item())
    losses.append(output.item())
    output.backward()
    optimizer.step()

preds= preds*np.std(train_theta,axis = 0)+np.mean(train_theta,axis = 0)
labels = training_labels.numpy()*np.std(train_theta,axis = 0)+np.mean(train_theta,axis = 0)

np.save('preds/labels_0',np.array(labels))
np.save('preds/preds_0', np.array(preds))
torch.save(model.state_dict(), 'preds/model')
torch.save(final_linear,'preds/linear')