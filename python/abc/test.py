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
import random
import matplotlib.pyplot as plt
random.seed(1408)
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
ls = np.random.uniform(low=0.0, high=0.1)
ls1 = np.random.uniform(low=0.0, high=0.1)



# Starting and true params for inference
current_model_params = options['current_model_params']
true_model_params = options['true_model_params']

# Set them as random uniform
current_model_params['lengthscale'] =  ls
true_model_params['lengthscale'] =  ls1

ber_params = options['ber_params']

# Generate "true" sequences under actual parameter set
real_sample, real_labels, real_seqs, real_full_g = gen_batch_with_seqs(
    germline, c_array, true_model_params, ber_params, t_batch_size, True
)

# Initialize network and optimizer
opt = optimizers.Adam(learning_rate=step_size)
autoencoder = build_nn(308)
autoencoder.compile(optimizer=opt, loss=custom_loss, metrics=[cond_variance])

# Define training generator
def genTraining(batch_size):
    while True:
        # Get training data for step
        mut, les = gen_nn_batch(
            germline, c_array, current_model_params, ber_params, batch_size, True
        )
        yield mut, les

# Generate test set under current model params.
t_batch_data, t_batch_labels = gen_nn_batch(
    germline, c_array, current_model_params, ber_params, t_batch_size
)
# Train Model
history = autoencoder.fit_generator(
    genTraining(batch_size),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(t_batch_data, t_batch_labels),
    verbose=2,
)

# Get predictions for real data
pred_labels = autoencoder.predict(real_sample)

# Variance for importance distribution is based on nn mse
#sampling_noise_sd = np.sqrt(history.history["cond_variance"][-1])
sampling_noise_sd = 0.00001
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
            "g": real_full_g[i],
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
w_temp = np.ones(n_imp_samples*t_batch_size)
est = cpf.lengthscale_inference(x_list, g_list, w_temp, np.linspace(0.00001,0.1,20), current_model_params)
true = true_model_params['lengthscale']
start = current_model_params['lengthscale']


def plot_index(n):
      real_label = real_labels[n//n_imp_samples]
      pred_label = pred_labels[n//n_imp_samples]
      x = x_list[n]
      g = g_list[n]
      w = w_list[n]
      ll = ll_list[n]
      q = q_list[n]
      fig = plt.figure()
      mut_index = np.where(np.array(real_seqs[n//n_imp_samples] != np.array(germline)))
      temp = {'real':real_label,'pred':pred_label,"x":x,'g':g,'w':w,'ll':ll,'q':q}
      # First plot, comparing pred g to true
      ax1 = fig.add_subplot(111)
      min_value = np.min(np.concatenate((temp['real'][:,0][temp['real'][:,0]!=0],temp['pred'][:,0][c_array==1],g_list[n])))
      max_value = np.max(np.concatenate((temp['real'][:,0][temp['real'][:,0]!=0],temp['pred'][:,0][c_array==1],g_list[n])))
      
      ax1.scatter(np.linspace(0,1,308)[temp['real'][:,0]!=0],temp['real'][:,0][temp['real'][:,0]!=0], s=10, c='b', marker="s", label='real g')
      ax1.scatter(np.linspace(0,1,308)[c_array==1],temp['pred'][:,0][c_array==1], s=10, c='r', marker="o", label='pred_g')
      ax1.scatter(x_list[n],g_list[n], s=10, c='m', marker="o", label='is g')
      ax1.vlines(np.linspace(0,1,308)[mut_index], ymin = min_value, ymax = max_value, colors = 'k')
      plt.legend(loc='best');
      return ax1

gp_kern_ridged = cpf.make_se_kernel(
              np.linspace(0,1,308), current_model_params['lengthscale'], current_model_params['gp_sigma'], current_model_params['gp_ridge']
          )
temp = []
for i in range(t_batch_size):      
          # P(G_k^i) in weights equation
      gp = np.log(
      scipy.stats.multivariate_normal.pdf(
            real_full_g[i], mean=current_model_params['gp_offset'] + np.zeros(308), cov=gp_kern_ridged)
        )
      temp.append(gp)
              