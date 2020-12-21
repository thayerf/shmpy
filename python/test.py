# Load our stuff
import numpy as np
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from nn_utils import gen_nn_batch, gen_batch_with_seqs
from nn_infrastructure import build_nn, custom_loss, cond_variance
from keras import optimizers
import cox_process_functions as cpf
# Path to germline sequence
germline_sequence = "data/gpt.fasta"
# Context model length and pos_mutating
context_model_length = 3
context_model_pos_mutating = 2
# Path to aid model
aid_context_model = "data/aid_logistic_3mer.csv"
# NN training params
batch_size = 100
num_epochs = 200
steps_per_epoch = 1
step_size = 0.005
t_batch_size = 300



germline = list(
    list(SeqIO.parse(germline_sequence, "fasta", alphabet=IUPAC.unambiguous_dna))[0].seq
)

n = np.size(germline)
c_array = np.zeros(n)
for i in range(n):
    c_array[i] = 1.0 * (germline[i] == "C")

ls = np.random.uniform(low = 0.0, high = 0.1)
# Starting and true params for inference
start_model_params = {
    "base_rate": 300.0,
    "lengthscale": 0.05,
    "gp_sigma": 10.0,
    "gp_ridge": 0.05,
    "gp_offset": -10,
}

true_model_params = { "base_rate" : 300.0,
                       "lengthscale" : ls,
                       "gp_sigma" : 10.0,
                       "gp_ridge" : .05,
                       "gp_offset": -10
}


ber_params = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}

opt = optimizers.Adam(learning_rate=step_size)
autoencoder = build_nn(308)
autoencoder.compile(optimizer=opt, loss=custom_loss, metrics=[cond_variance])


def genTraining(batch_size):
    while True:
        # Get training data for step
        mut, les = gen_nn_batch(
            germline, c_array, start_model_params, ber_params, batch_size
        )
        yield mut, les


t_batch_data, t_batch_labels = gen_nn_batch(
    germline, c_array, start_model_params, ber_params, t_batch_size
)

real_sample, real_labels, real_seqs  = gen_batch_with_seqs(germline,c_array,true_model_params,ber_params, t_batch_size)

history = autoencoder.fit_generator(
    genTraining(batch_size),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(t_batch_data, t_batch_labels),
    verbose=2,
)


sampling_noise_sd = np.sqrt(history.history['cond_variance'][-1])

pred_labels = autoencoder.predict(real_sample)

current_model_params = start_model_params
n_imp_samples = 200
sampling_noise_sd = np.sqrt(history.history['cond_variance'][-1])
x_list = []
g_list = []
g_true = []
w_list = []
for i in range(t_batch_size):
    for j in range(n_imp_samples):
          
        complete_data = {"A": pred_labels[i,:,1], "A_tilde": pred_labels[i,:,2], "g": pred_labels[i,:,0], "S": germline, "S_1": real_seqs[i]}
        imp_sam = cpf.complete_data_sample_around_cond_means_sgcp(c_array,complete_data, current_model_params,ber_params, sampling_noise_sd)
        x_list.append(np.append(imp_sam["A"], imp_sam["A_tilde"]))
        g_list.append(imp_sam["g"])
        w_list.append(imp_sam["w"])
        g_true.append(complete_data["g"])