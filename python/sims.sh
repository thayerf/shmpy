#!/bin/bash
source ~/.bashrc
conda activate mmr
num_sims=$1
for (( i=1;i<=$1;i++ ))
do
srun python test_mmr.py > output.log &
done

echo "$num_sims sims submitted"
