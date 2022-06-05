import logging
from collections import defaultdict
from copy import deepcopy

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
        self.round_limit = 65

    def sample(self, round_idx, candidates, client_num_per_round):
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        num_clients = min(client_num_per_round * 10, len(candidates))
        indexes = np.random.choice(candidates, num_clients, replace=False)
        times = map(self.get_client_completion_time, indexes)
        client_indexes = []
        for i, time in enumerate(times):
            if len(client_indexes) >= client_num_per_round:
                break
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
            avail_weight = self.calc_avail_weight(client, init_weight, round_idx, max_pen)
            hw_weight = self.calc_hw_weight(client, init_weight, round_idx)
            weight = (avail_weight + hw_weight) / 2
            if self.args.score_method == 'mul':
                weight = np.sqrt(avail_weight * hw_weight)
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / np.sum(weights)
        return weights

    def calc_avail_weight(self, client, init_weight, round_idx, max_pen):
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
        return init_weight

    def calc_hw_weight(self, client, init_weight, round_idx):
        if self.client_times[client] == 0:
            return init_weight
        return 1 - self.client_times[client] / self.args.round_timeout


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

    def __init__(self, aggregator_args, model_size, train_num_dict, test_for_selected_clients) -> None:
        super().__init__(aggregator_args, model_size, train_num_dict)
        self.tier_count = 5
        self.selected_tier = 0
        self.update_interval = int(self.args.comm_round * .02)
        self.tiers = [[] for _ in range(self.tier_count)]
        self.credits = self.create_credits()
        self.probabilities = [1 / self.tier_count] * self.tier_count
        self.old_tiers_acc = []
        self.tiers_acc = []
        self.assign_clients_to_tiers()
        self.test_for_selected_clients = test_for_selected_clients

    def create_credits(self):
        logging.debug('START: create credits')
        step = 1.5
        base = self.args.comm_round / sum(step ** i for i in range(self.tier_count))
        credits = [int(np.round(base * (step ** i))) for i in range(self.tier_count - 1)]
        credits.append(self.args.comm_round - sum(credits))
        credits.reverse()
        logging.debug('END: create credits')
        return credits

    def assign_clients_to_tiers(self):
        logging.debug('START: assign clients')
        clients = [index for index in range(self.args.client_num_in_total)]
        self.tiers = np.array_split(np.argsort(list(map(self.get_client_completion_time, clients))), self.tier_count)
        logging.debug('END: assign clients')

    def calc_tiers_accuracy(self):
        logging.debug('START: calc accuracies')
        self.tiers_acc = []
        for t in self.tiers:
            self.tiers_acc.append(self.test_for_selected_clients(t))
        logging.debug('END: calc accuracies')

    def sample(self, round_idx, candidates, client_num_per_round):
        if round_idx % self.update_interval == 0 and round_idx >= self.update_interval:
            self.calc_tiers_accuracy()
            if self.old_tiers_acc and self.tiers_acc[self.selected_tier] <= self.old_tiers_acc[self.selected_tier]:
                self.update_probabilities()
            self.old_tiers_acc = deepcopy(self.tiers_acc)

        logging.debug('START: select tier')
        self.selected_tier = np.random.choice(range(len(self.tiers)), 1, replace=False, p=self.probabilities)[0]
        logging.info('TiFL selected tier for round {} = {}'.format(round_idx, self.selected_tier))
        self.credits[self.selected_tier] -= 1
        logging.debug('END: select tier')

        logging.debug('START: select clients')
        candidates = list(set(candidates).intersection(self.tiers[self.selected_tier]))
        num_clients = min(client_num_per_round, len(candidates))
        client_indexes = np.random.choice(candidates, num_clients, replace=False)
        logging.debug('END: select clients')

        if self.credits[self.selected_tier] == 0:
            logging.info('TiFL tier {} of {} is out of credits.'.format(self.selected_tier, len(self.tiers)))

            logging.debug('START: removing tier')
            self.tiers.pop(self.selected_tier)
            self.credits.pop(self.selected_tier)
            self.probabilities.pop(self.selected_tier)
            if self.old_tiers_acc:
                self.old_tiers_acc.pop(self.selected_tier)
            if self.tiers_acc:
                self.tiers_acc.pop(self.selected_tier)
            logging.debug('END: removing tier')
            self.update_probabilities()
            self.selected_tier = 0
        return client_indexes

    def update_probabilities(self):
        logging.debug('START: update probs')
        n = len(self.tiers)
        d = n * (n - 1) / 2
        if self.tiers_acc:
            a = np.argsort(self.tiers_acc)
            for i, tier in enumerate(a):
                self.probabilities[tier] = (n - i) / d
            self.probabilities = list(np.array(self.probabilities) / sum(self.probabilities))
        else:
            self.probabilities = [1 / n] * n
        logging.debug('END: update probs')
