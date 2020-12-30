#!/bin/bash
source ~/.bashrc
conda activate keras_tf
num_sims=$1
for (( i=1;i<=$1;i++ ))
do
srun python test.py > output.log &
done

echo "$num_sims sims submitted"
