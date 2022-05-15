import argparse
from scipy import stats
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

from fedml_core.availability.simulation import load_sim_data


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

    parser.add_argument(
        "--partition_alpha", type=float, default=0.5, metavar="PA", help="partition alpha (default: 0.5)"
    )

    parser.add_argument(
        "--client_num_in_total", type=int, default=3400, metavar="NN",
        help="number of workers in a distributed cluster"
    )

    parser.add_argument("--client_num_per_round", type=int, default=3, metavar="NN", help="number of workers")

    parser.add_argument(
        "--batch_size", type=int, default=20, metavar="N", help="input batch size for training (default: 64)"
    )

    parser.add_argument("--client_optimizer", type=str, default="sgd", help="SGD with momentum; adam")

    parser.add_argument("--backend", type=str, default="MPI", help="Backend for Server and Client")

    parser.add_argument("--lr", type=float, default=0.03, metavar="LR", help="learning rate (default: 0.001)")

    parser.add_argument("--wd", help="weight decay parameter;", type=float, default=0.0001)

    parser.add_argument("--epochs", type=int, default=1, metavar="EP", help="how many epochs will be trained locally")

    parser.add_argument("--comm_round", type=int, default=3, help="how many round of communications we shoud use")

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
sim_data = load_sim_data(args)

data_analysis = []
for data in sim_data:
    tr = data.trace
    active = False
    if tr['active'][0] == 0:
        tr['active'].pop(0)
        active = True
    time = 0
    total_time = tr['finish_time']
    status_change = len(tr['active'])
    active_durations = []
    inactive_durations = []
    while True:
        if not active:
            arr = tr['active']
            dur_arr = inactive_durations
        else:
            arr = tr['inactive']
            dur_arr = active_durations
        if len(arr) == 0:
            dur_arr.append(total_time - time)
            break

        change_time = arr.pop(0)
        dur_arr.append(change_time - time)
        time = change_time
        active = not active
    data_analysis.append({
        'total_time': total_time,
        'active_durations': active_durations,
        'status_change': status_change / total_time,
        'inactive_durations': inactive_durations,
        'comp_time': data.get_completion_time(85140.453125)
    })

df = pd.DataFrame(data_analysis)
# df['active_percent'] = df['active_durations'].apply(sum) / df['total_time']
# max_time = df['comp_time'].max()
# df['mean_active'] = df['active_durations'].apply(lambda durations: np.array(durations).mean())
#
# df = df[['active_percent', 'mean_active', 'status_change']]
# df_n = (df - df.min()) / (df.max() - df.min())
# df_n['status_change'] = 1 - df_n['status_change']
# df_n['total_score'] = df_n.mean(axis=1)
#
# worst_to_best = df_n['total_score'].to_numpy().argsort()
# np.save('avail_worst_to_best.npy', worst_to_best)

# for c in df.columns:
#     plt.figure()
#     label = c
#     data = df[c]
#     if 'durations' in c:
#         label = 'mean_' + c[:-1]
#         data = df[c].apply(np.mean)
#     plt.hist(data, rwidth=0.9)
#     plt.xlabel(label)
#     plt.ylabel('number of clients')
#     plt.show()
