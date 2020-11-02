# Load our stuff
import pkgutil
from pathlib import Path
import numpy as np
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from SHMModels.fitted_models import ContextModel
from keras import optimizers, Input, Model
from keras.models import Sequential
import keras.backend as K
import matplotlib.pyplot as plt
from keras.layers import (
    Dense,
    TimeDistributed,
    SimpleRNN,
    Dropout,
    Conv2D,
    Conv2DTranspose,
    Flatten,
    Conv1D,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
)
import scipy
import scipy.stats
import math
from imp_sam import *
from minisim import *
import sys

##### USER INPUTS (Edit some of these to be CLI eventually)

# Path to germline sequence
germline_sequence = "data/gpt.fasta"
# Context model length and pos_mutating
context_model_length = 3
context_model_pos_mutating = 2
# Path to aid model
aid_context_model = "data/aid_logistic_3mer.csv"
# Num seqs and n_mutation rounds
n_seqs = 1
n_mutation_rounds = 1
# step size
step_size = 0.01
# batch size num epochs
batch_size = 500
num_epochs = 500
steps_per_epoch = 1
gl1 = list(list(
    SeqIO.parse(germline_sequence, "fasta", alphabet=IUPAC.unambiguous_dna)
)[0].seq)
germline = list(list(
    SeqIO.parse(germline_sequence, "fasta", alphabet=IUPAC.unambiguous_dna)
)[0].seq)

start = np.random.uniform(high=0.05)
true = np.random.uniform(high=0.05)

# .01, .025
start_model_params = { "base_rate" : 0.25,
                       "lengthscale" : start,
                       "gp_sigma" : 10.0,
                       "gp_ridge" : .1,
                       "gp_offset" : 12.0
}
true_model_params = { "base_rate" : 0.25,
                       "lengthscale" : true,
                       "gp_sigma" : 2.0,
                       "gp_ridge" : .1,
                       "gp_offset": 12.0
}
print(start_model_params)
print(true_model_params)

n = np.size(germline)
c_array = np.zeros(n)
for i in range(n):
    c_array[i] = 1.0 * (germline[i] == 'C')
vals = np.random.poisson(lam = true_model_params['base_rate'], size = n)
vals = np.multiply(vals, c_array)
unit = []
for i in range(np.size(vals)):
    if vals[i] > 0:
        conts = (i-np.random.uniform(size = int(vals[i])))/n
        unit = np.append(unit, conts)
        
        
# We want a custom loss that only penalizes GP at lesion values
def custom_loss(y_true, y_pred):
    g_true = y_true[:,:,0]
    g_pred = y_pred[:,:,0]
    ind = K.cast((y_true[:,:,1]+y_true[:,:,2]) > 0, 'float32')
    g_adj = (g_true-g_pred)*ind
    return K.mean(K.square(K.stack((g_adj,y_true[:,:,1]-y_pred[:,:,1],y_true[:,:,2]-y_pred[:,:,2]))))


# Let's build our encoder. Seq is of length 308.
input_seq = Input(shape=(308, 4, 1))

# We add 2 convolutional layers.
x = Conv2D(16, (3, 6), activation="relu", padding="same")(input_seq)
x = MaxPooling2D((2, 1), padding="same")(x)
x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)

# Now we decode back up
x = Conv2DTranspose(
    filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
)(x)
x = Conv2DTranspose(
    filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
)(x)
x = Conv2DTranspose(
    filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu"
)(x)
x = Flatten()(x)
# I think ReLU is fine here because the values are nonnegative?
decoded = Dense(units=(308*3), activation="linear")(x)
reshaped = Reshape((308,3))(decoded)

# at this point the "decoded" representation is a 3*308 vector indicating our predicted # of
#lesions, prelesions, gp values at each site.
autoencoder = Model(input_seq, reshaped)
autoencoder.compile(optimizer="adam", loss=custom_loss)

print(autoencoder.summary(90))

def genTraining(batch_size):
    while True:
        # Get training data for step
        mut,les = gen_batch(germline,start_model_params,batch_size,c_array)
        yield mut,les
t_batch_data, t_batch_labels = gen_batch(germline,start_model_params, 1000,c_array)
# Train
history = autoencoder.fit_generator(
    genTraining(batch_size),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(t_batch_data, t_batch_labels),
    verbose=2,
)


real_sample, real_labels  = gen_batch(germline,true_model_params, 300,c_array)
pred_labels = autoencoder.predict(real_sample)

current_model_params = start_model_params
n_imp_samples = 200
sampling_noise_sd = 0.1
x_list = []
g_list = []
w_list = []
ll_list= []
for i in range(300):
    for j in range(n_imp_samples):
        # Sample lesions and prelesions, and get full GP values
        new_A = np.multiply(c_array,np.random.poisson(lam = np.maximum(pred_labels[i,:,1],0)))
        new_A_tilde = np.multiply(c_array,np.random.poisson(lam = np.maximum(pred_labels[i,:,2],0)))
        g_full = pred_labels[i,:,0]
        # Now, we want to include only lesion sites on unit interval, as well as GP values for those sites
        g = []
        A_unit = []
        A_tilde_unit = []
        for p in range(308):
            if new_A[p] >0 :
                for q in range(int(new_A[p])):
                    g=np.append(g,g_full[p])
                    A_unit=np.append(A_unit, discrete_to_interval(p,308))
        for p in range(308):
            if new_A_tilde[p] >0 :
                for q in range(int(new_A_tilde[p])):
                    g= np.append(g,g_full[p])
                    A_tilde_unit= np.append(A_tilde_unit,discrete_to_interval(p,308))
        
                    
        
        complete_data = {"A": A_unit, "A_tilde": A_tilde_unit, "g": g}
        imp_sam,pois,thinning, gp= complete_data_sample_around_true_states_sgcp(complete_data = complete_data,
                                                                   params = current_model_params,
                                                                   sampling_sd = sampling_noise_sd)
        ll_list.append([pois,thinning, gp])
        x_list.append(np.append(complete_data["A"], imp_sam["A_tilde"]))
        g_list.append(imp_sam["g"])
        w_list.append(imp_sam["w"])
l_test = np.linspace(.001, .999, 50)
grid,probs = lengthscale_inference(x_list, g_list, w_list, l_test_grid = l_test, model_params = current_model_params, full_grid = True)
print(start)
print(grid[np.argmax(probs)])
print(true)
file1 = open("big/start","a")
file2 = open("big/first","a")
file3 = open("big/true","a")
temp = [str(start),","]
file1.writelines(temp)
temp = [str(grid[np.argmax(probs)]),","]
file2.writelines(temp)
temp = [str(true),","]
file3.writelines(temp)
file1.close()
file2.close()
file3.close()
