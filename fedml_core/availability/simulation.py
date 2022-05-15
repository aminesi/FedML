import os
import pickle
from abc import ABC, abstractmethod
import numpy as np


class BaseSim(ABC):
    @abstractmethod
    def is_active(self, cur_time):
        pass

    @abstractmethod
    def get_completion_time(self, model_size):
        pass

    def active_till_the_end(self, cur_time, model_size):
        pass


class ClientSim(BaseSim):
    def __init__(self, trace, speed, args):
        self.trace = trace
        self.compute_speed = speed['computation']
        self.bandwidth = speed['communication']
        self.args = args
        self.behavior_index = 0

    def is_active(self, cur_time):

        norm_time = cur_time % self.trace['finish_time']

        while norm_time > self.trace['inactive'][self.behavior_index]:
            if self.behavior_index == 0 and norm_time > self.trace['inactive'][-1]:
                break
            self.behavior_index += 1
            self.behavior_index %= len(self.trace['active'])

        if self.trace['active'][self.behavior_index] <= norm_time <= self.trace['inactive'][self.behavior_index]:
            return True

        return False

    def active_till_the_end(self, cur_time, model_size):
        if not self.is_active(cur_time):
            return False
        end_time = cur_time + self.get_completion_time(model_size)
        norm_time = cur_time % self.trace['finish_time']
        end_norm = end_time % self.trace['finish_time']
        if end_norm < norm_time:
            end_norm = end_norm + self.trace['finish_time']
        if end_norm <= self.trace['inactive'][self.behavior_index]:
            return True
        return False

    def get_completion_time(self, model_size):
        return 3 * self.args.batch_size * self.args.epochs * float(self.compute_speed) / 1000 \
               + 2 * model_size / float(self.bandwidth)


def load_sim_data(aggregator_args):
    script_dir = os.path.dirname(__file__)
    with open(os.path.join(script_dir, 'client_behave_trace'), 'rb') as tr:
        trace_data = list(pickle.load(tr).values())

    with open(os.path.join(script_dir, 'client_device_capacity'), 'rb') as cp:
        capacity_data = list(pickle.load(cp).values())

    worst_to_best = list(np.load(os.path.join(script_dir, 'avail_worst_to_best.npy')))
    client_sim_data = []

    if aggregator_args.trace_distro == 'random':
        indices = range(aggregator_args.client_num_in_total)
    elif aggregator_args.trace_distro == 'high_avail':
        indices = distribute(.6, .2, aggregator_args, worst_to_best)
    elif aggregator_args.trace_distro == 'low_avail':
        indices = distribute(.2, .6, aggregator_args, worst_to_best)
    elif aggregator_args.trace_distro == 'average':
        indices = distribute(.2, .2, aggregator_args, worst_to_best)
    else:
        raise AttributeError(
            'Invalid trace_distro. Possible options: {}'.format('"random" or "high_avail" or "low_avail" or "average"'))

    for client_id in indices:
        client_sim = ClientSim(
            trace_data[client_id % len(trace_data)],
            capacity_data[client_id % len(capacity_data)],
            aggregator_args
        )
        client_sim_data.append(client_sim)
    return client_sim_data


def distribute(high, low, args, worst_to_best):
    best_count = int(np.round(high * args.client_num_in_total))
    worst_count = int(np.round(low * args.client_num_in_total))
    mid_count = args.client_num_in_total - best_count - worst_count
    start = int(np.round(len(worst_to_best) / 2 - mid_count / 2))
    return worst_to_best[:worst_count] + worst_to_best[-best_count:] + worst_to_best[start:start + mid_count]
