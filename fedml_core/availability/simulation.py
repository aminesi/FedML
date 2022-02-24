import os
import pickle
from abc import ABC, abstractmethod


class BaseSim(ABC):
    @abstractmethod
    def is_active(self, cur_time):
        pass

    @abstractmethod
    def get_completion_time(self, model_size):
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

        if norm_time > self.trace['inactive'][self.behavior_index]:
            self.behavior_index += 1

        self.behavior_index %= len(self.trace['active'])

        if self.trace['active'][self.behavior_index] <= norm_time <= self.trace['inactive'][self.behavior_index]:
            return True

        return False

    def get_completion_time(self, model_size):
        return 3 * self.args.batch_size * self.args.epochs * float(self.compute_speed) / 1000 \
               + 2 * model_size / float(self.bandwidth)


def load_sim_data(aggregator_args):
    script_dir = os.path.dirname(__file__)
    with open(os.path.join(script_dir, 'client_behave_trace'), 'rb') as tr:
        trace_data = pickle.load(tr)

    with open(os.path.join(script_dir, 'client_device_capacity'), 'rb') as cp:
        capacity_data = pickle.load(cp)

    client_sim_data = []
    for client_id in range(1, aggregator_args.client_num_in_total + 1):
        client_sim = ClientSim(
            trace_data[client_id % len(trace_data)],
            capacity_data[client_id % len(capacity_data)],
            aggregator_args
        )
        client_sim_data.append(client_sim)
    return client_sim_data
