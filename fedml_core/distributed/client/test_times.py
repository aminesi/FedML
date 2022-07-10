import argparse
from scipy import stats
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

from fedml_api.distributed.fedavg.client_selector import RandomSelector, FedCs, TiFL, MdaSelector
from fedml_core.availability.base_selector import BaseSelector
from fedml_core.availability.simulation import load_sim_data

import logging

from mpi4py import MPI

i = MPI.COMM_WORLD.Get_rank()

prefix = 'Worker {}'.format(i) if i != 0 else 'Server'

logging.basicConfig(
    level=logging.NOTSET,
    format="%(asctime)s  %(levelname)s  (%(filename)s:%(lineno)d)  " + prefix + " -  %(message)s",
    datefmt="%H:%M:%S",
)


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument("--model", type=str, default="cnn", metavar="N", help="neural network used in training")

    parser.add_argument("--dataset", type=str, default="mnist", metavar="N", help="dataset used for training")

    parser.add_argument("--data_dir", type=str, default="./../../../data/FederatedEMNIST", help="data directory")

    parser.add_argument(
        "--partition_method",
        type=str,
        default="hetero",
        metavar="N",
        help="how to partition the dataset on local workers",
    )
    parser.add_argument('--time_mode', type=str, default='simulated')  # "none" or "simulated"
    parser.add_argument('--selector', type=str, default='random')  # "random" or "fedcs" or "tifl" or "tiflx" or "mda"
    parser.add_argument('--checkpoints', nargs='+', type=int, default=[])
    parser.add_argument('--allow_failed_clients', type=str, default='no')  # 'yes' or 'no'
    parser.add_argument('--trace_distro', type=str,
                        default='average')  # "random" or "high_avail" or "low_avail" or "average"
    parser.add_argument('--round_timeout', type=int, default=860)
    parser.add_argument('--fedcs_time', type=int, default=300)
    parser.add_argument('--score_method', type=str, default='mul')  # "add" or "mul"
    parser.add_argument('--mda_method', type=str, default='avail')  # "avail" or "mix"
    parser.add_argument('--tifl_mode', type=str, default='prob')
    # Oort params
    parser.add_argument(
        "--partition_alpha", type=float, default=0.5, metavar="PA", help="partition alpha (default: 0.5)"
    )

    parser.add_argument(
        "--client_num_in_total", type=int, default=500, metavar="NN",
        help="number of workers in a distributed cluster"
    )

    parser.add_argument("--client_num_per_round", type=int, default=10, metavar="NN", help="number of workers")

    parser.add_argument(
        "--batch_size", type=int, default=20, metavar="N", help="input batch size for training (default: 64)"
    )

    parser.add_argument("--client_optimizer", type=str, default="sgd", help="SGD with momentum; adam")

    parser.add_argument("--backend", type=str, default="MPI", help="Backend for Server and Client")

    parser.add_argument("--lr", type=float, default=0.03, metavar="LR", help="learning rate (default: 0.001)")

    parser.add_argument("--wd", help="weight decay parameter;", type=float, default=0.0001)

    parser.add_argument("--epochs", type=int, default=1, metavar="EP", help="how many epochs will be trained locally")

    parser.add_argument("--comm_round", type=int, default=2500, help="how many round of communications we shoud use")

    parser.add_argument(
        "--is_mobile", type=int, default=1, help="whether the program is running on the FedML-Mobile server side"
    )

    parser.add_argument("--frequency_of_the_test", type=int, default=1, help="the frequency of the algorithms")

    parser.add_argument("--gpu_server_num", type=int, default=1, help="gpu_server_num")

    parser.add_argument("--gpu_num_per_server", type=int, default=4, help="gpu_num_per_server")

    parser.add_argument(
        "--gpu_mapping_file",
        type=str,
        default="gpu_mapping.yaml",
        help="the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.",
    )

    parser.add_argument(
        "--gpu_mapping_key", type=str, default="mapping_default", help="the key in gpu utilization file"
    )

    parser.add_argument(
        "--grpc_ipconfig_path",
        type=str,
        default="grpc_ipconfig.csv",
        help="config table containing ipv4 address of grpc server",
    )

    parser.add_argument(
        "--trpc_master_config_path",
        type=str,
        default="trpc_master_config.csv",
        help="config indicating ip address and port of the master (rank 0) node",
    )

    parser.add_argument("--ci", type=int, default=0, help="CI")
    args = parser.parse_args()
    return args


parser = argparse.ArgumentParser()
args = add_args(parser)
logging.getLogger().setLevel(logging.ERROR)

for s in ['random', 'mda', 'fedcs', 'tifl', 'tiflx']:
    args.selector = s
    logging.error('\n\n{}'.format(s))

    if s == 'random':
        client_selector = RandomSelector(args, 427386.5234375, 0)
    elif s == 'mda':
        client_selector = MdaSelector(args, 427386.5234375, 0)
    elif s == 'fedcs':
        client_selector = FedCs(args, 427386.5234375, 0)
    elif s == 'tifl' or s == 'tiflx':
        client_selector = TiFL(args, 427386.5234375, 0, None)
    else:
        raise AttributeError('Unknown clients selector. selector can be "random" or "fedcs" or "oort"')

    client_selector.simulate()


# random
# 01:44:00  ERROR  (base_selector.py:111)  Server -  participants num: 492
# 01:44:00  ERROR  (base_selector.py:112)  Server -  failed rounds: 1039
# 01:44:00  ERROR  (base_selector.py:113)  Server -  2500: 1709592.0781860352
# 01:44:00  ERROR  (test_times.py:132)  Server -
#
# mda
# 01:44:21  ERROR  (base_selector.py:111)  Server -  participants num: 481
# 01:44:21  ERROR  (base_selector.py:112)  Server -  failed rounds: 676
# 01:44:21  ERROR  (base_selector.py:113)  Server -  2500: 1616575.1212921143
# 01:44:21  ERROR  (test_times.py:132)  Server -
#
# fedcs
# 01:44:24  ERROR  (base_selector.py:111)  Server -  participants num: 362
# 01:44:24  ERROR  (base_selector.py:112)  Server -  failed rounds: 562
# 01:44:24  ERROR  (base_selector.py:113)  Server -  2500: 928120.4834098816
# 01:44:24  ERROR  (test_times.py:132)  Server -
#
# tifl
# 01:44:25  DEBUG  (client_selector.py:148)  Server -  START: create credits
# 01:44:25  DEBUG  (client_selector.py:154)  Server -  END: create credits
# 01:44:25  DEBUG  (client_selector.py:158)  Server -  START: assign clients
# 01:44:25  DEBUG  (client_selector.py:161)  Server -  END: assign clients
# 01:44:27  ERROR  (base_selector.py:111)  Server -  participants num: 493
# 01:44:27  ERROR  (base_selector.py:112)  Server -  failed rounds: 711
# 01:44:27  ERROR  (base_selector.py:113)  Server -  2500: 877288.4594078064
# 01:44:27  ERROR  (test_times.py:132)  Server -
#
# tiflx
# 01:44:28  DEBUG  (client_selector.py:148)  Server -  START: create credits
# 01:44:28  DEBUG  (client_selector.py:154)  Server -  END: create credits
# 01:44:28  DEBUG  (client_selector.py:158)  Server -  START: assign clients
# 01:44:28  DEBUG  (client_selector.py:161)  Server -  END: assign clients
# 01:44:35  ERROR  (base_selector.py:111)  Server -  participants num: 483
# 01:44:35  ERROR  (base_selector.py:112)  Server -  failed rounds: 529
# 01:44:35  ERROR  (base_selector.py:113)  Server -  2500: 768188.370803833