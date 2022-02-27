import numpy as np

from fedml_core.availability.base_selector import BaseSelector, TimeMode


class RandomSelector(BaseSelector):

    # noinspection PyMethodMayBeStatic
    def sample(self, round_idx, candidates, client_num_per_round):
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(candidates, client_num_per_round, replace=False)
        return client_indexes


class FedCs(BaseSelector):
    def __init__(self, model_size, train_num_dict, time_mode=TimeMode.NONE) -> None:
        super().__init__(model_size, train_num_dict, time_mode)
        self.round_limit = 6

    def sample(self, round_idx, candidates, client_num_per_round):
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        indexes = np.random.choice(candidates, client_num_per_round, replace=False)
        times = map(self.get_client_completion_time, indexes)
        client_indexes = []
        for i, time in enumerate(times):
            if time < self.round_limit:
                client_indexes.append(indexes[i])
        return client_indexes
