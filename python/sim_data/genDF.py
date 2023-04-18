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
random.seed(1408)
import csv
import collections
# Load options
import pandas as pd
import glob
from random import sample
# Get df with all seqs
df = pd.read_csv('../data/shmoof_edges_11-Jan-2023_NoNode0_iqtree_K80+R.csv')
bases = ['A','C','G','T']
# Remove the indels from the sequence pairs
for i in range(len(df)):
    index = np.array([df['orig_seq'][i][j] in bases and  df['mut_seq'][i][j] in bases for j in range(len(df['orig_seq'][i]))])
    parent_string = ""
    child_string = ""
    parent_array = [df['orig_seq'][i][j] for j in np.where(index)[0]]
    parent_string = parent_string.join(parent_array)
    child_array = [df['mut_seq'][i][j] for j in np.where(index)[0]]
    child_string = child_string.join(child_array)
    df.loc[i,'orig_seq'] = parent_string
    df.loc[i,'mut_seq'] = child_string

df.to_pickle('data/full_edge_df.pk1')
