# Load our stuff
import numpy as np
from Bio import SeqIO
from nn_utils import gen_nn_batch, gen_batch_with_seqs
from nn_infrastructure import build_nn, custom_loss, cond_variance
from keras import optimizers
import cox_process_functions as cpf
import logging
import os
import sys
import json
from tensorflow import keras
import keras.backend as K
# Load options
with open(sys.argv[1], "r") as read_file:
    options = json.load(read_file)

# We want to suppress the logging from tf
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['KMP_WARNINGS'] = '0'
# Path to germline sequence
germline_sequence = "data/gpt.fasta"
# Context model length and pos_mutating
context_model_length = 3
context_model_pos_mutating = 2
# Path to aid model
aid_context_model = "data/aid_logistic_3mer.csv"
# NN training params
batch_size = options['batch_size']
num_epochs = options['num_epochs']
step_size = options['step_size']
t_batch_size = options['t_batch_size']
steps_per_epoch = 1

# n_imp_sam
n_imp_samples = options['n_imp_samples']


germline = list(
    list(SeqIO.parse(germline_sequence, "fasta"))[0].seq
)

n = np.size(germline)
c_array = np.zeros(n)
for i in range(n):
    c_array[i] = 1.0 * (germline[i] == "C")
    
# Sample lengthscales from prior
ls = 0.005235300536761945
ls1 = 0.03805807905341564


# Starting and true params for inference
current_model_params = options['current_model_params']
true_model_params = options['true_model_params']

# Set them as random uniform
current_model_params['lengthscale'] =  ls
true_model_params['lengthscale'] =  ls1

ber_params = options['ber_params']

# Generate "true" sequences under actual parameter set
real_sample, real_labels, real_seqs = gen_batch_with_seqs(
    germline, c_array, true_model_params, ber_params, t_batch_size
)


# Initialize network and optimizer
autoencoder = keras.models.load_model('model.pt', custom_objects = {'custom_loss': custom_loss, 'cond_variance':cond_variance, 'K':K})


# Get predictions for real data
pred_labels = autoencoder.predict(real_sample)

# Variance for importance distribution is based on nn mse
sampling_noise_sd = np.sqrt(19.100)
# Initialize list trackers
x_list = []
g_list = []
g_true = []
w_list = []
q_list = []
ll_list = []
for i in range(t_batch_size):
    for j in range(n_imp_samples):

        complete_data = {
            "A": pred_labels[i, :, 1],
            "A_tilde": pred_labels[i, :, 2],
            "g": pred_labels[i, :, 0],
            "S": germline,
            "S_1": real_seqs[i],
        }
        imp_sam = cpf.complete_data_sample_around_cond_means_sgcp(
            c_array, complete_data, current_model_params, ber_params, sampling_noise_sd
        )
        x_list.append(np.append(imp_sam["A"], imp_sam["A_tilde"]))
        g_list.append(imp_sam["g"])
        w_list.append(imp_sam["w"])
        g_true.append(complete_data["g"])
        q_list.append([imp_sam["q1"],imp_sam["q2"],imp_sam["q3"]])
        ll_list.append(imp_sam['ll_list'])
est = cpf.lengthscale_inference(x_list, g_list, w_list, np.linspace(0.00001,0.1,25), current_model_params)
true = true_model_params['lengthscale']
start = current_model_params['lengthscale']


