from enum import IntEnum
import numpy as np
from scipy.stats import gumbel_r


class ALInput(IntEnum):
    TRAIN_PRED = 0
    VAL_PRED = 1
    TEST_PRED = 2

    TRAIN_LABEL = 3
    VAL_LABEL = 4
    TEST_LABEL = 5

    TRAIN_LOSS = 6
    VAL_LOSS = 7
    TEST_LOSS = 8

    TRAIN_EMBEDDING = 9
    VAL_EMBEDDING = 10
    TEST_EMBEDDING = 11


def has_train_input(input_types):
    """
    Returns True if any input type to active learning strategy needs information on the training set.
    Otherwise, return False.

    :param Set input_types: Set of ALInput indicating the necessary information needed for a strategy as input.
    :return: True if any training input type is in input_types. False otherwise.
    """
    for input in input_types:
        if input % 3 == 1:
            return True
    return False


def has_val_input(input_types):
    """
    Returns True if any input type to active learning strategy needs information on the validation set.
    Otherwise, return False.

    :param Set input_types: Set of ALInput indicating the necessary information needed for a strategy as input.
    :return: True if any validation input type is in input_types. False otherwise.
    """
    for input in input_types:
        if input % 3 == 2:
            return True
    return False


def has_test_input(input_types):
    """
    Returns True if any input type to active learning strategy needs information on the test set.
    Otherwise, return False.

    :param Set input_types: Set of ALInput indicating the necessary information needed for a strategy as input.
    :return: True if any testing input type is in input_types. False otherwise.
    """
    for input in input_types:
        if input % 3 == 0:
            return True
    return False


# List of available active learning strategies.
strategies = {}


class Strategy:
    """
    Abstract class for active learning strategies.
    """
    def __init__(self, strategy_config, trainer):
        """
        :param Dict strategy_config: dictionary mapping from parameter name to value.
        :param ALDataset dataset: ALDataset object containing all of training/validation/testing data and embeddings.
        """
        self.strategy_config = strategy_config
        self.trainer = trainer
        self.dataset = trainer.dataset
        self.input_types = None

    def __init_subclass__(cls, **kwargs):
        """
        Register strategy by its strategy_name.
        """
        super().__init_subclass__(**kwargs)
        strategies[cls.strategy_name] = cls

    def select(self, trainer, budget):
        """
        Selecting a batch of examples based on output from the trainer.

        :param Trainer trainer: trainer object with most up-to-date model.
        :param int budget: total batch size budget for the selection.
        :return: A list of indexes of the original dataset indexing the batch for annotation.
        """
        pass

    def get_softmax_samples(self, scores_N, beta, aquisition_batch_size):
        '''
        Adapted from https://arxiv.org/pdf/2106.12059
        '''
        noised_scores_N = np.array(scores_N) + gumbel_r.rvs(loc=0, scale=1 / beta, size=len(scores_N), random_state=None)
        return np.argsort(noised_scores_N)[-aquisition_batch_size:]
        
    def get_power_samples(self, scores_N, beta, aquisition_batch_size):
        '''
        Adapted from https://arxiv.org/pdf/2106.12059
        '''
        return self.get_softmax_samples(np.log(np.array(scores_N)), beta=beta, aquisition_batch_size=aquisition_batch_size)
        
    def get_softrank_samples(self, scores_N, beta, aquisition_batch_size):
        '''
        Adapted from https://arxiv.org/pdf/2106.12059
        '''
        sorted_indices_N = np.argsort(-np.array(scores_N))
        ranks_N = np.argsort(sorted_indices_N) + 1
        return self.get_power_samples(1 / ranks_N, beta=beta, aquisition_batch_size=aquisition_batch_size)
