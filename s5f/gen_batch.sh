#!/bin/bash
source ~/.bashrc
conda activate keras_tf

for (( i=0;i<=1000;i++ ))
do
	srun -t 0-3:00 -q restart-new -p restart-new python -u s5f.py &
done