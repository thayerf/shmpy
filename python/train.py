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
# Set data path (CLI?)
data_path = 'sim_data/'

# Load test and train data
train_X = np.loadtxt(data_path+'train_X', delimiter= ',')
test_X = np.loadtxt(data_path+'test_X', delimiter= ',')
train_theta = np.loadtxt(data_path+'train_theta', delimiter= ',')
test_theta = np.loadtxt(data_path+'test_theta', delimiter= ',')

# Get length of training data
train_n = np.shape(train_X)[0]

# Define training generator
def genTraining(batch_size):
    while True:
        # Get training data for step
        ind = random.sample(range(train_n), batch_size)
        yield (
            train_X[ind,:],
            [
                train_theta[ind,:]
            ],
        )

# Define our NN
model = keras.Sequential()
model.add(keras.layers.Dense(64, input_dim= 105, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(6, activation='linear'))
hist = keras.callbacks.History()
# Give summary of architecture

# Initialize optimizer with given step size
adam = keras.optimizers.Adam(lr = 0.05)
# Compile model w/ pinball loss, use mse as metric
model.compile(
    loss=['mse'],
    optimizer=adam,
)

print(model.summary(90))
# Train the model using generator and callbacks we defined w/ test set as validation

history = model.fit_generator(
    genTraining(500),
    epochs=10,
    steps_per_epoch=1,
    callbacks=[hist],
    validation_data=(test_X, test_theta)
)

pred_labels = model.predict(test_X)
np.save('labels_0',np.array(test_theta))
np.save('preds_0', np.array(pred_labels))
model.save("model")
