from enum import Enum
from typing import List

import numpy as np
import logging
from fedml_core.availability.simulation import BaseSim, load_sim_data


class TimeMode(Enum):
    # REAL = 1
    SIMULATED = 1
    NONE = 2


class BaseSelector:

    def __init__(self, model_size, train_num_dict, time_mode=TimeMode.NONE) -> None:
        self.args = None
        self.client_times = None

        self.time_mode = time_mode
        self.client_sim_data: List[BaseSim] = []
        self.selected_clients = []
        self.cur_time = -1
        self.model_size = model_size
        self.train_num_dict = train_num_dict

    def initialize(self, aggregator_args):
        self.args = aggregator_args
        self.client_times = np.array([0] * self.args.client_num_in_total).astype(np.float32)

        if self.time_mode == TimeMode.SIMULATED:
            self.client_sim_data = load_sim_data(self.args)

    def is_client_active(self, client_id, time):
        if self.time_mode != TimeMode.SIMULATED:
            return True
        return self.client_sim_data[client_id].is_active(time)

    def get_client_completion_time(self, client_id):
        if self.time_mode == TimeMode.NONE:
            return 0

        return self.client_sim_data[client_id].get_completion_time(self.model_size)

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if self.cur_time == -1:
            self.cur_time = 0
        else:
            self.cur_time += np.max(self.client_times[self.selected_clients])
        candidates = [i for i in range(client_num_in_total) if self.is_client_active(i, self.cur_time)]
        if len(candidates) <= client_num_per_round:
            self.selected_clients = candidates
        else:
            self.selected_clients = self.sample(round_idx, candidates, client_num_per_round)
        logging.info('Current time is: {}'.format(self.cur_time))
        logging.info('Sampled clients for round {}: {}'.format(round_idx, self.selected_clients))
        return self.selected_clients

    # noinspection PyMethodMayBeStatic
    def sample(self, round_idx, candidates, client_num_per_round):
        pass
