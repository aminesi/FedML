import pickle
import sys
from abc import ABC
from enum import Enum
from typing import List
import logging
import os

import numpy as np

from fedml_core.availability.simulation import load_sim_data, BaseSim


class TimeMode(Enum):
    # REAL = 1
    SIMULATED = 1
    NONE = 2


class BaseAggregator(ABC):

    def __init__(self, worker_num, args, time_mode: TimeMode = TimeMode.NONE):
        self.time_mode = time_mode
        self.client_sim_data: List[BaseSim] = []
        self.args = args

        self.worker_num = worker_num

        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        self.client_times = np.array([0] * self.args.client_num_in_total)
        self.selected_clients = []

        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.cur_time = -1
        if time_mode == TimeMode.SIMULATED:
            self.client_sim_data = load_sim_data(self.args)

    def is_client_active(self, client_id, time):
        if self.time_mode != TimeMode.SIMULATED:
            return True
        return self.client_sim_data[client_id].is_active(time)

    def get_client_completion_time(self, client_id, model_size):
        if self.time_mode == TimeMode.NONE:
            return 0

        return self.client_sim_data[client_id].get_completion_time(model_size)

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

    def sample(self, round_idx, candidates, client_num_per_round):
        pass

    def aggregate(self):
        pass

    def add_local_trained_result(self, worker_index, model_params, sample_num):
        logging.info('add_model. index = %d' % worker_index)
        self.model_dict[worker_index] = model_params
        self.sample_num_dict[worker_index] = sample_num
        self.flag_client_model_uploaded_dict[worker_index] = True

        model_size = sys.getsizeof(pickle.dumps(model_params)) / 1024.0 * 8
        client_id = self.selected_clients[worker_index]
        self.client_times[client_id] = self.get_client_completion_time(client_id, model_size)
        logging.info('Aggregator: client {} finished in {} seconds'.format(client_id, self.client_times[client_id]))

    def check_whether_all_receive(self):
        logging.debug('worker_num = {}'.format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

# TODO handle availability change mid training
