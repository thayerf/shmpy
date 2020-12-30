#!/bin/bash
num_sims=$1
counter = 1
while [ $counter -le num_sims ]
do
srun python test.py > output.log &
((counter++))
done

echo "$num_sims sims submitted"