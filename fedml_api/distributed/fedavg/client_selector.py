import logging

import numpy as np

from fedml_api.distributed.fedavg.oort import OortHelper
from fedml_core.availability.base_selector import BaseSelector, TimeMode


class RandomSelector(BaseSelector):

    # noinspection PyMethodMayBeStatic
    def sample(self, round_idx, candidates, client_num_per_round):
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        num_clients = min(client_num_per_round, len(candidates))
        client_indexes = np.random.choice(candidates, num_clients, replace=False)
        return client_indexes


class FedCs(BaseSelector):

    def __init__(self, aggregator_args, model_size, train_num_dict, time_mode=TimeMode.NONE) -> None:
        super().__init__(aggregator_args, model_size, train_num_dict, time_mode)
        self.round_limit = 6

    def sample(self, round_idx, candidates, client_num_per_round):
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        num_clients = min(client_num_per_round, len(candidates))
        indexes = np.random.choice(candidates, num_clients, replace=False)
        times = map(self.get_client_completion_time, indexes)
        client_indexes = []
        for i, time in enumerate(times):
            if time < self.round_limit:
                client_indexes.append(indexes[i])
        return client_indexes


class Oort(BaseSelector):

    def __init__(self, aggregator_args, model_size, train_num_dict, time_mode=TimeMode.NONE) -> None:
        super().__init__(aggregator_args, model_size, train_num_dict, time_mode)
        self.helper = OortHelper(self.args)
        for client_id in range(self.args.client_num_in_total):
            feedbacks = {'reward': min(self.train_num_dict[client_id], self.args.epochs * self.args.batch_size),
                         'duration': self.get_client_completion_time(client_id)}
            self.helper.register_client(client_id, feedbacks)

    def sample(self, round_idx, candidates, client_num_per_round):
        if round_idx != 0:
            self.update_oort_helper(round_idx)
        num_clients = min(client_num_per_round, len(candidates))
        client_indexes = self.helper.select_participant(num_clients, candidates)
        return client_indexes

    def update_oort_helper(self, round_idx):
        for client in self.selected_clients:
            self.helper.update_client_util(client, {
                'reward': self.clients_training_metrics[client]['oort_score'],
                'duration': self.client_times[client],
                'status': True,
                'time_stamp': round_idx
            })


class TiFL(BaseSelector):

    def __init__(self, aggregator_args, model_size, train_num_dict, time_mode=TimeMode.NONE) -> None:
        super().__init__(aggregator_args, model_size, train_num_dict, time_mode)
        self.tier_count = 5
        self.tiers = [[] for _ in range(self.tier_count)]
        self.selected_tier = 0
        self.credits = [100] * self.tier_count
        self.probabilities = [1 / self.tier_count] * self.tier_count
        self.update_interval = 50
        self.old_tiers_acc = None
        self.tiers_acc = None
        self.assign_clients_to_tiers()

    def assign_clients_to_tiers(self):
        clients = [index for index in range(self.args.client_num_in_total)]
        self.tiers = np.array_split(np.argsort(map(self.get_client_completion_time, clients)), self.tier_count)

    def calc_tiers_accuracy(self):
        # todo do get accuracy on test
        accuracies = [0.0] * self.tier_count
        self.old_tiers_acc = self.tiers_acc
        self.tiers_acc = accuracies

    def sample(self, round_idx, candidates, client_num_per_round):
        self.calc_tiers_accuracy()
        if round_idx % self.update_interval == 0 and round_idx >= self.update_interval:
            if self.tiers_acc[self.selected_tier] <= self.old_tiers_acc[self.selected_tier]:
                self.update_probabilities()

        self.selected_tier = -1
        while self.selected_tier != -1:
            selected_tier = np.random.choice(range(self.tier_count), 1, replace=False, p=self.probabilities)[0]
            if self.credits[selected_tier] > 0:
                self.credits[selected_tier] -= 1
                self.selected_tier = selected_tier

        candidates = list(set(candidates).intersection(self.tiers[self.selected_tier]))
        num_clients = min(client_num_per_round, len(candidates))
        client_indexes = np.random.choice(candidates, num_clients, replace=False)
        return client_indexes

    def update_probabilities(self):
        # todo the approach in paper is not correct
        # n = np.sum(np.array(self.credits) > 0)
        # d = n * (n - 1) / 2
        # a = np.argsort(self.tiers_acc)
        # for i, tier in enumerate(a):
        #     self.probabilities[tier] = (n-i)/d
        pass
