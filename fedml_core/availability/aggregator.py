import pickle
from abc import ABC
from types import Union
from enum import Enum
from typing import List


class TimeMode(Enum):
    # REAL = 1
    SIMULATED = 1
    NONE = 2


class ClientSim:
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


class BaseAggregator(ABC):

    def __init__(self, args, time_mode: TimeMode = TimeMode.NONE):
        self.time_mode = time_mode
        self.client_sim_data: List[ClientSim] = []
        self.args = args

        if time_mode == TimeMode.SIMULATED:
            self.load_sim_data()

    def load_sim_data(self):
        with open('client_behave_trace', 'rb') as tr:
            trace_data = pickle.load(tr)

        with open('client_device_capacity', 'rb') as cp:
            capacity_data = pickle.load(cp)

        for client_id in range(self.args.client_num_in_total):
            client_sim = ClientSim(
                trace_data[client_id % len(trace_data)],
                capacity_data[client_id % len(capacity_data)],
                self.args
            )
            self.client_sim_data.append(client_sim)

    def is_client_active(self, client_id, cur_time):
        if self.time_mode != TimeMode.SIMULATED:
            return True
        return self.client_sim_data[client_id].is_active(cur_time)

    def get_client_completion_time(self, client_id, model_size):
        if self.time_mode == TimeMode.NONE:
            return 0

        return self.client_sim_data[client_id].get_completion_time(model_size)

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        pass

    def aggregate(self):
        pass

    def check_whether_all_receive(self):
        pass

    def add_local_trained_result(self, index, model_params, sample_num):
        pass
