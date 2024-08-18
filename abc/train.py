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
with open("./sim_data/data.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        data.append(row)
data = [s for s in data if len(s) == 112]
data = np.array(data)

train_X = data[data[:,111]==0][:,0:105]
train_theta = data[data[:,111]==0][:,105:111]
test_X = data[data[:,111]==1][:,0:105]
test_theta = data[data[:,111]==1][:,105:111]

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

# Define our NN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_dim= 105, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(12, activation='linear'))
# Give summary of architecture
def meanAndVariance(y_true,y_pred) :
  """Loss function that has the values of the last axis in y_true 
  approximate the mean and variance of each value in the last axis of y_pred."""
  mean = y_pred[..., 0::2]
  variance = y_pred[..., 1::2]
  res = tf.math.square(mean - y_true) + tf.math.square(variance - tf.math.square(mean - y_true))
  return tf.math.reduce_mean(res, axis=-1)
# Initialize optimizer with given step size
adam = tf.keras.optimizers.Adam(learning_rate = lr)
# Compile model w/ pinball loss, use mse as metric
model.compile(
    loss=[meanAndVariance],
    optimizer=adam,
)

log_dir = "logs/fits/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# Train the model using generator and callbacks we defined w/ test set as validation

history = model.fit(
    genTraining(1000),
    epochs=40000,
    steps_per_epoch=1,
    callbacks=[tensorboard_callback],
    validation_data=(test_X, test_cent)
)

pred_labels = (model.predict(test_X)*std) + mean
np.save('preds/labels_10',np.array(test_theta))
np.save('preds/preds_10', np.array(pred_labels))
model.save("preds/model10")
