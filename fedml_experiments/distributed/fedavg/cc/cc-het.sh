#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH hetjob
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

cd ~/FedML


source venv/bin/activate

cd ./fedml_experiments/distributed/fedavg/
./cc/cc_run_with_conf_het cifar


