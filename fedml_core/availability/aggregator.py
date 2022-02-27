import logging
import pickle
import sys
from abc import ABC
from enum import Enum

from fedml_core.availability.base_selector import BaseSelector


class BaseAggregator(ABC):

    def __init__(self, worker_num, args, client_selector: BaseSelector):

        self.args = args
        self.client_selector = client_selector
        self.client_selector.initialize(args)
        self.worker_num = worker_num

        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        return self.client_selector.client_sampling(round_idx, client_num_in_total, client_num_per_round)

    def aggregate(self):
        pass

    def add_local_trained_result(self, worker_index, model_params, sample_num):
        logging.info('add_model. index = %d' % worker_index)
        self.model_dict[worker_index] = model_params
        self.sample_num_dict[worker_index] = sample_num
        self.flag_client_model_uploaded_dict[worker_index] = True

        client_id = self.client_selector.selected_clients[worker_index]
        self.client_selector.client_times[client_id] = self.client_selector.get_client_completion_time(client_id)
        logging.info('Aggregator: client {} finished in {} seconds'.format(client_id, self.client_selector.client_times[
            client_id]))

    def check_whether_all_receive(self):
        for idx in range(len(self.client_selector.selected_clients)):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(len(self.client_selector.selected_clients)):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

# TODO handle availability change mid training
