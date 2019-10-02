from SHMModels.simulate_mutations import simulate_sequences_abc
from SHMModels.simulate_mutations import memory_simulator
# We would like to simulate sequences in place, with the fasta file and context model loaded into memory
# This way we can simulate A LOT of training data without writing to disk (slow)
import numpy as np
from scipy.special import logit
from sklearn.preprocessing import scale

def hot_encode_2d(seq):
      seq_hot = np.zeros((len(seq),4,1))
      for j in range(len(seq)):
                  seq_hot[j,0,0]= (seq[j] == 'A')
                  seq_hot[j,1,0]= (seq[j] == 'T')
                  seq_hot[j,2,0]= (seq[j] == 'G')
                  seq_hot[j,3,0]= (seq[j] == 'C')
      return(seq_hot)
      
      
def gen_batch_2d(batch_size):
      params, seqs = memory_simulator(sequence, aid_model, n_seqs,n_mutation_rounds, n_sims = batch_size)
      seqs = seqs[:,0]
      seqs = [i.decode('utf-8') for i in seqs]
      seqs = [list(i) for i in seqs]
      seqs_hot = np.zeros((len(seqs)//n_seqs,n_seqs,len(seqs[1]),9,1))
      for i in range(len(seqs)):
            for j in range(len(seqs[1])):
                  seqs_hot[i//n_seqs,i%n_seqs,j,0,0]= (seqs[i][j] == 'A')
                  seqs_hot[i//n_seqs,i%n_seqs,j,1,0]= (seqs[i][j] == 'T')
                  seqs_hot[i//n_seqs,i%n_seqs,j,2,0]= (seqs[i][j] == 'G')
                  seqs_hot[i//n_seqs,i%n_seqs,j,3,0]= (seqs[i][j] == 'C')
      orig_seq_rep = np.repeat(orig_seq, n_seqs*batch_size, axis = 2)
      orig_seq_rep = (np.moveaxis(orig_seq_rep,-1,0))
      orig_seq_rep = orig_seq_rep.reshape((batch_size,n_seqs,308,4,1))
      seqs_hot[:,:,:,4:8,:] = orig_seq_rep
      seqs_hot[:,:,:,8,:] = (seqs_hot[:,:,:,0:4,:] == seqs_hot[:,:,:,4:8,:]).all(axis = 3)
      params[:,4:8]= logit(params[:,4:8])
      params = (params - means)/sds
      return({"seqs":seqs_hot, "params":params})
# Create iterator for simulation
def genTraining_2d(batch_size):
    while True:
        # Get training data for step
        dat = gen_batch_2d(batch_size)
        # We repeat the labels for each x in the sequence
        batch_labels = dat['params']
        batch_data = dat['seqs']
        yield batch_data,batch_labels
  
def gen_batch_1d(batch_size):
      params, seqs = memory_simulator(sequence, aid_model, n_seqs,n_mutation_rounds, n_sims = batch_size)
      seqs = seqs[:,0]
      seqs = [i.decode('utf-8') for i in seqs]
      seqs = [list(i) for i in seqs]
      seqs_hot = np.zeros((len(seqs),4*len(seqs[1])))

      for i in range(len(seqs)):
            for j in range(len(seqs[1])):
                  seqs_hot[i,4*j]= (seqs[i][j] == 'A')
                  seqs_hot[i,(4*j+1)]= (seqs[i][j] == 'T')
                  seqs_hot[i,(4*j+2)]= (seqs[i][j] == 'G')
                  seqs_hot[i,(4*j+3)]= (seqs[i][j] == 'C')
      params[:,4:8]= logit(params[:,4:8])
      params = (params - means)/sds
      params_ext = np.repeat(params,n_seqs, axis = 0)
      return({"seqs":seqs_hot, "params":params_ext})
# Create iterator for simulation
def genTraining_1d(batch_size):
    while True:
        # Get training data for step
        dat = gen_batch_1d(batch_size)
        # We repeat the labels for each x in the sequence
        batch_labels = dat['params']
        batch_data = dat['seqs']
        yield batch_data,batch_labels
  
  

