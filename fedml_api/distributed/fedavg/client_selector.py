import logging
from collections import defaultdict

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

    def __init__(self, aggregator_args, model_size, train_num_dict) -> None:
        super().__init__(aggregator_args, model_size, train_num_dict)
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


class MdaSelector(BaseSelector):

    # factors:
    # 1. failure history
    # 2. expected continues availability

    def __init__(self, aggregator_args, model_size, train_num_dict) -> None:
        super().__init__(aggregator_args, model_size, train_num_dict)
        self.failure_history = defaultdict(list)
        self.availability_history = [[] for _ in range(self.args.client_num_in_total)]
        self.init_round_time = []

    def sample(self, round_idx, candidates, client_num_per_round):
        self.init_round_time.append(self.cur_time)
        available = set(candidates)

        for i in range(self.args.client_num_in_total):
            self.availability_history[i].append(i in available)

        if round_idx > 0:
            for client in self.failed_clients:
                self.failure_history[client].append(round_idx - 1)

        ws = self.calculate_weights(candidates, round_idx)
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        num_clients = min(client_num_per_round, len(candidates))
        client_indexes = np.random.choice(candidates, num_clients, replace=False, p=ws)
        return client_indexes

    def calculate_weights(self, candidates, round_idx):
        max_pen = 1
        if round_idx > 0:
            max_pen = sum(1 / i for i in range(1, round_idx + 1))
        weights = []
        for client in candidates:
            init_weight = .5
            failure_history = np.array(self.failure_history[client])
            availability_history = list(zip(self.init_round_time, self.availability_history[client]))
            if len(availability_history) > 10:
                availability_history = availability_history[-10:]
                last_time, last_available = 0, False
                total_active = 0
                start_time = -1
                for time, is_available in availability_history:
                    if start_time == -1:
                        start_time = time
                    if is_available and last_available:
                        total_active += (time - last_time)
                    last_available = is_available
                    last_time = time

                active_percentage = total_active / (self.cur_time - start_time)
                init_weight += (active_percentage - 0.5) * 2 * init_weight

            if len(failure_history) > 0:
                penalty = (1 / (round_idx - failure_history)).sum()
                init_weight *= (1 - penalty / max_pen)
            weights.append(init_weight)

        weights = np.array(weights)
        weights = weights / np.sum(weights)
        return weights


class Oort(BaseSelector):

    def __init__(self, aggregator_args, model_size, train_num_dict) -> None:
        super().__init__(aggregator_args, model_size, train_num_dict)
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

    def __init__(self, aggregator_args, model_size, train_num_dict) -> None:
        super().__init__(aggregator_args, model_size, train_num_dict)
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
