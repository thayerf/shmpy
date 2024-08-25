# shmpy
Fitting mechanistic parameters requires installing the mechanistic forward simulator: https://github.com/thayerf/SHMModels

Simulating and fitting the Neural Network ABC estimator consists of the following steps:

1. Download the data and move it to `python/data`. Data must be a csv with columns `orig_seq` and `mut_seq`
2. `cd abc/sim_data`
3. Run `genDF.py` to filter sequences you put in `python/data`.
4. Run `gen_batch.sh` to simulate 10,000 summary statistics parameter pairs.
5. `cd ..`
6. Run `train.py` to fit the model to these pairs.

The `abc/variable_importance`, `attention`, and `s5f` folders are used to generate plots for the paper.
