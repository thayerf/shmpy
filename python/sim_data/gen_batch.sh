#!/bin/bash

for (( i=0;i<=99;i++ ))
do
	srun -t 0-3:00 -q restart-new -p restart-new python -u genDat.py &
done