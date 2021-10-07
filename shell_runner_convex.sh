#!/bin/bash

# Usage of src.experiments_runner
# Arg1: real or synth
#    real => will run real dataset experiments (a9a, quantum, phishing, superconduct, w8a)
#    synth => will run synthetic dataset (logistic/linear regression)
# Arg2: dataset
#    if Arg1 = real, must be: a9a, quantum, phishing, superconduct, w8a
#    if Arg1 = synth, must be: linear or logistic
# Arg3: kind of algos that will be used for the run.
#    either uni-vs-bi, either mcm-vs-existing
# Arg4: iid ('idd' or 'non-iid')
#    if Arg1 = synth, this argument is useless

# To kill all jobs: kill -9 `cat save_pid.txt` && rm save_pid.txt

algos=mcm-vs-existing

# If not existing, creates a logs repository.
[ ! -d "./logs" ] && mkdir ./logs
[ -d "save_pid.txt" ] && rm save_pid.txt

for dataset in logistic linear
do
  nohup python3 -m src.experiments_runner synth $dataset $algos NONE > logs/${dataset}.txt & 2>&1 &
  echo $! >> save_pid.txt
done

for iid in iid non-iid
do
	for dataset in a9a quantum phishing superconduct w8a
	do
	  nohup python3 -m src.experiments_runner real $dataset $algos $iid > logs/log_${dataset}_${iid}.txt 2>&1 &
	  echo $! >> save_pid.txt
	done
done