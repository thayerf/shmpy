# shmpy

Simulating and fitting the Neural Network ABC estimator consists of 3 steps:

1. Run `sim_data/genDF.py` to filter sequences. This requires a csv with the columns 'orig_seq' and 'mut_seq'
2. Run `sim_data/gen_batch.sh` to simulate 10,000 summary statistics parameter pairs
3. Run train.py to fit the model to these pairs. 

The variable importance and s5f folders are used to generate plots for the paper.
