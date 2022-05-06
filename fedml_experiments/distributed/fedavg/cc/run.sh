#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out
#SBATCH --ntasks=11
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=p100:1
#SBATCH --mem-per-cpu=4G
#SBATCH --gpu-bind=single:1

cd ~/scratch/FedML


source venv/bin/activate

cd ./fedml_experiments/distributed/fedavg/
./cc/cc_run_with_conf femnist
