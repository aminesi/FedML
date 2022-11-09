# FL real-world challenges simulation 

This repository contains the code and results for the paper "MDA: Availability-Aware Federated Learning Client Selection" submitted to IEEE Transactions on Reliability journal.

## Replication

### Main experiments
This is an extension of the popular framework [FedML](https://github.com/FedML-AI/FedML) that adds the support for multiple client selection techniques and simulation of client availability and resource heterogeneity from real-world datasets.

All experiments are done using python 3.8 and PyTorch 1.10.

Steps to run the experiments are as follows:

1. Install the required dependencies using ``pip`` and the requirement file located at `./requirements-cc.txt` if not already installed.
   
2. Run the main program by ```./fedml_experiments/distributed/fedavg/run_with_conf {config_file_name}```. There are a few notes regarding config files:

   * The `config_file_name` can be `cifar` or `femnist` to use available config files for ***CIFAR-10*** and ***FEMNIST*** datasets.
   - The available config files are located at `./fedml_experiments/distributed/fedavg/configs` and any new config file should be added in that directory.
   - config files should be ***.json*** files.

### Config details

To replicate different experiments, the following options in the available config files can be changed: 

```yaml
    "trace_distro": one of the following 
      "high_avail"
      "low_avail"
      "average"
    "selector": one of the following 
      "random"
      "fedcs"
      "tifl"
      "mda"
      "tiflx" # this is the proposed TiFL-MDA from the paper
```

Notes:
1. For the experiments to run the root folder must be named ***FedML***.
2. Other options are available for configuration files to further customize the experiments. Their description can be found in the file `./fedml_experiments/distributed/fedavg/main_fedavg.py`.
3. The experiments run in a distributed manner and rely on MPI and the system must be capable of running multiple processes using `mpirun`. 


### Results Visualization

Results can be visualized using the Jupiter notebooks available at `./fedml_experiments/distributed/fedavg/`.

* `visualize.ipynb`, shows all the results together for each dataset, whereas `latex-prep.ipynb` generates and stores the results like shown in the paper.
* These notebook files must be run from the `./fedml_experiments/distributed/fedavg/` directory.



## Results 

All the raw results are available at `./fedml_experiments/distributed/fedavg/results/{cifar|femnist}`.

All the processed results are available at `./fedml_experiments/distributed/fedavg/results/latex`