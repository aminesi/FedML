#!/bin/bash

#SBATCH --time=07:00:00
#SBATCH --output=%x.out
#SBATCH --ntasks=11
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=v100l:1
#SBATCH --mem-per-cpu=2G
#SBATCH --gpu-bind=single:1

cd ~/scratch/FedML

module load python/3.8 StdEnv/2020  gcc/10.3.0  openmpi/4.1.1 mpi4py/3.1.3
source venv/bin/activate

cd ./fedml_experiments/distributed/fedavg/
./cc/cc_run_with_conf femnist