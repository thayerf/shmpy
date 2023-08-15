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
from Bio.Alphabet.IUPAC import unambiguous_dna, ambiguous_dna
from random import sample
# Get df with all seqs
path = '.../edge_dfs' # use your path
all_files = glob.glob(path + "/*")


li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)
bases = ['A','C','G','T']
# Remove the indels from the sequence pairs
for i in range(len(df)):
    index = np.array([df['PARENT_SEQ'][i][j] in bases and  df['CHILD_SEQ'][i][j] in bases for j in range(len(df['PARENT_SEQ'][i]))])
    parent_string = ""
    child_string = ""
    parent_array = [df['PARENT_SEQ'][i][j] for j in np.where(index)[0]]
    parent_string = parent_string.join(parent_array)
    child_array = [df['CHILD_SEQ'][i][j] for j in np.where(index)[0]]
    child_string = child_string.join(child_array)
    df.loc[i,'PARENT_SEQ'] = parent_string
    df.loc[i,'CHILD_SEQ'] = child_string

df.to_pickle('full_edge_df.pk1')