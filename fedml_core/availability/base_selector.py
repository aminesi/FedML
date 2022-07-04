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

    def __init__(self, aggregator_args, model_size, train_num_dict) -> None:
        self.args = aggregator_args
        self.client_times = np.array([0] * self.args.client_num_in_total).astype(np.float32)

        self.time_mode = TimeMode.NONE
        if self.args.time_mode != 'none':
            self.time_mode = TimeMode.SIMULATED
        self.client_sim_data: List[BaseSim] = []
        self.selected_clients = []
        self.failed_clients = []
        self.clients_training_metrics = {}
        self.cur_time = -1
        self.model_size = model_size
        self.train_num_dict = train_num_dict
        self.round_timeout = self.args.round_timeout
        self.times = []

        if self.time_mode == TimeMode.SIMULATED:
            self.client_sim_data = load_sim_data(self.args)

    def is_client_active(self, client_id, time):
        if self.time_mode != TimeMode.SIMULATED:
            return True
        return self.client_sim_data[client_id].is_active(time)

    def is_client_active_till_the_end(self, client_id, time):
        if self.time_mode != TimeMode.SIMULATED:
            return True
        return self.client_sim_data[client_id].active_till_the_end(time, self.model_size)

    def get_client_completion_time(self, client_id):
        if self.time_mode == TimeMode.NONE:
            return 0

        return self.client_sim_data[client_id].get_completion_time(self.model_size)

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if self.cur_time == -1:
            self.cur_time = 0
        else:
            if len(self.selected_clients) == 0 or len(self.failed_clients) > 0:
                self.cur_time += self.round_timeout
            else:
                self.cur_time += np.max(self.client_times[self.selected_clients])
            self.times.append(self.cur_time)
        candidates = [i for i in range(client_num_in_total) if self.is_client_active(i, self.cur_time)]
        self.selected_clients = self.sample(round_idx, candidates, client_num_per_round)

        if self.args.allow_failed_clients == 'no':
            new_clients = list(filter(lambda client: self.is_client_active_till_the_end(client, self.cur_time) and
                                                     self.get_client_completion_time(client) <= self.round_timeout,
                                      self.selected_clients))
            self.failed_clients = list(set(self.selected_clients).difference(new_clients))
            self.selected_clients = new_clients

        logging.info('Current time is: {}'.format(self.cur_time))
        logging.info('Sampled clients for round {}: {}'.format(round_idx, self.selected_clients))
        logging.info('Round {} failed clients: {}'.format(round_idx, self.failed_clients))

        return self.selected_clients

    # noinspection PyMethodMayBeStatic
    def sample(self, round_idx, candidates, client_num_per_round):
        return []

    def finish(self):
        if len(self.selected_clients) == 0 or len(self.failed_clients) > 0:
            self.cur_time += self.round_timeout
        else:
            self.cur_time += np.max(self.client_times[self.selected_clients])
        self.times.append(self.cur_time)
        np.save(self.args.output_dir + 'times.npy', np.array(self.times))

    def simulate(self):
        all = []
        for i in range(self.args.comm_round):
            logging.getLogger().setLevel(logging.ERROR)
            self.client_sampling(i, self.args.client_num_in_total, self.args.client_num_per_round)
            all += self.selected_clients
            for client_id in self.selected_clients:
                self.client_times[client_id] = self.get_client_completion_time(client_id)
        if len(self.selected_clients) == 0 or len(self.failed_clients) > 0:
            self.cur_time += self.round_timeout
        else:
            self.cur_time += np.max(self.client_times[self.selected_clients])
        self.times.append(self.cur_time)
        print(len(set(all)))
        logging.error('{}: {}'.format(self.args.comm_round, self.cur_time))
