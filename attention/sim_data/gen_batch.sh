#!/bin/bash

for (( i=0;i<=99;i++ ))
do
	srun -t 0-3:00  python -u genDat.py &
done