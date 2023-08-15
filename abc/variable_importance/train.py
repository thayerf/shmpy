# Load our stuff
import tensorflow as tf
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
import datetime
from random import sample
import csv
# Set lr
lr = 10e-6
# Load test and train data
data = []
with open("../sim_data/data/data.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        data.append(row)
print(len(data))
data = [s for s in data if len(s) == 112]
print(len(data))
data = np.array(data)

train_X = data[data[:,111]==0][:,0:105]
train_theta = data[data[:,111]==0][:,105:111]
test_X = data[data[:,111]==1][:,0:105]
test_theta = data[data[:,111]==1][:,105:111]


x_mean = np.mean(train_X, axis =0)
x_sd = np.std(train_X, axis =0)
mean = np.mean(train_theta, axis =0)
std = np.std(train_theta, axis =0)


x_mean = np.mean(train_X, axis =0)
x_sd = np.std(train_X, axis =0)
mean = np.mean(train_theta, axis =0)
std = np.std(train_theta, axis =0)
# Center labels
train_cent = (train_theta - mean)/std
test_cent = (test_theta - mean)/std
# Center training data
train_X = (train_X-x_mean)/x_sd
test_X =  (test_X - x_mean)/x_sd
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
                train_cent[ind,:]
            ],
        )


losses = []
# Replace each variable with gaussian noise and fit the model
for i in range(105):
    # Define our NN
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim= 105, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(6, activation='linear'))
    # Give summary of architecture

    # Initialize optimizer with given step size
    adam = tf.keras.optimizers.Adam(learning_rate = lr)
    # Compile model w/ MSE loss
    model.compile(
        loss=['mse'],
        optimizer=adam,
    )
    # Train the model using generator and callbacks we defined w/ test set as validation
    noise_out_train = train_X[:,i]
    noise_out_test = test_X[:,i] 
    train_X[:,i] = np.random.normal(size=len(train_X[:,i]))
    test_X[:,i] = np.random.normal(size=len(test_X[:,i]))
    history = model.fit(
        genTraining(500),
        epochs=20000,
        steps_per_epoch=1,
        validation_data=(test_X, test_cent),
        verbose = 0
    )
    val_loss = history.history['val_loss'][-1]
    losses.append(val_loss)
    train_X[:,i] = noise_out_train
    test_X[:,i] = noise_out_test
    print(i)
    print(val_loss)
np.savetxt('losses', np.array(losses))
