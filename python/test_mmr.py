# Load our stuff
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from SHMModels.simulate_mutations import MutationRound
from SHMModels.fitted_models import ContextModel
import pkgutil
import logging
import os
import sys
import json
import random
import matplotlib.pyplot as plt
random.seed(1408)
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


def sample_prior():
    ls = np.random.uniform(low = -12.0, high = -2.0)
    sg = np.random.uniform(low = 5.0, high = 15.0)
    br = np.random.uniform(low = 0.05, high = 0.25)
    off = -10
    p_fw = np.random.uniform(low = 0.0,high = 1.0)
    return { "base_rate" : br,
                       "lengthscale" : np.exp(ls),
                       "gp_sigma" : sg,
                       "gp_ridge" : .01,
            "gp_offset": off,
            "p_fw": p_fw
            }

# Get batch
def gen_batch_letters(seq,batch_size, params):
       # The prior specification
    ber_lambda = 0.5
    ber_params = [0.25,0.25,0.25,0.25]
    
    bubble_size = 25.0
    exo_left = 0.2
    exo_right = 0.2
    pol_eta_params = {
        "A": [0.9, 0.02, 0.02, 0.06],
        "G": [0.01, 0.97, 0.01, 0.01],
        "C": [0.01, 0.01, 0.97, 0.01],
        "T": [0.06, 0.02, 0.02, 0.9],
    }
    prior_params = sample_prior()
    mutated_seq_list = []
    for i in range(batch_size):
          mr = MutationRound(
          seq,
          ber_lambda=ber_lambda,
          mmr_lambda=1 - ber_lambda,
          replication_time=100,
          bubble_size=bubble_size,
          aid_time=10,
          exo_params={"left": exo_left, "right": exo_right},
          pol_eta_params=pol_eta_params,
          ber_params=ber_params,
          p_fw=prior_params['p_fw'],
          aid_context_model=cm,
          log_ls = prior_params['lengthscale'],
          br = prior_params['base_rate'],
          sg = prior_params['gp_sigma'],
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

def get_mmr_summ(seqs, germline):
    c_mut_count = 0
    g_mut_count = 0
    for i in seqs:
        c_mut_count += np.sum(np.logical_and(i != np.array(list(germline)),np.array([j == 'C' for j in germline])))
        g_mut_count += np.sum(np.logical_and(i != np.array(list(germline)),np.array([j == 'G' for j in germline])))
    return c_mut_count/(c_mut_count+g_mut_count)

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
    for i in range(n_imp_samp):
        
        model_params = sample_prior()
        sample = gen_batch_letters(germline, n, model_params)
        
        sample_bp = (1.0-np.mean(sample == np.array(list(germline)), axis = 0))
        
        sample_colocals = get_colocal(sample,germline,sample_bp,50)
        sample_mmr_stat = get_mmr_summ(sample, germline)
        colocal = np.append(sample_colocals[0:50:5], np.mean(sample_bp))
        colocal = np.append(colocal, sample_mmr_stat)
        w_list.append(gauss_kernel(colocal, base_colocal,eps))
        ls_list.append(model_params['lengthscale'])
        sg_list.append(model_params['gp_sigma'])
        rate_list.append(model_params['base_rate'])
        p_fw_list.apppend(model_params['p_fw'])
        if i % 50 == 0:
            print(i)
    return rate_list, ls_list, sg_list, p_fw_list,  w_list, base_colocal

true_model_params = sample_prior()
obs_sample = gen_batch_letters(germline, 1000, true_model_params)

rate_list, ls_list, sg_list, p_fw_list,  w_list, base_colocal = importance_sample(obs_sample,germline, 1000, 500, 2.0)



pred_mean_ls = np.dot(w_list,ls_list)/np.sum(w_list)
pred_mean_sig = np.dot(w_list, sg_list)/np.sum(w_list)
pred_mean_rate = np.dot(w_list, rate_list)/np.sum(w_list)
pred_mean_p_fw = np.dot(w_list, p_fw_list)/np.sum(w_list)
true_ls = true_model_params['lengthscale']
true_sig = true_model_params['gp_sigma']
true_rate = true_model_params['base_rate']
true_p_fw = true_model_params['p_fw']


f = open("est_ls", "a")
f.write(str(pred_mean_ls) + " ")
f.close()

f = open("est_rate","a")
f.write(str(pred_mean_rate) + " ")
f.close()

f = open('est_sig', 'a')
f.write(str(pred_mean_sig) + " ")
f.close()

f = open('est_p_fw', 'a')
f.write(str(pred_mean_p_fw) + " ")
f.close()

f = open("true_ls", "a")
f.write(str(true_ls) + " ")
f.close()

f = open('true_rate', 'a')
f.write(str(true_rate) + " ")
f.close()

f = open('true_sig', 'a')
f.write(str(true_sig) + ' ')
f.close()

f = open('true_p_fw', 'a')
f.write(str(true_p_fw) + " ")
f.close()


