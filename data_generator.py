import numpy as np
from logger import Logger


class GenerateData:
    def __init__(self, config):
        self.config = config
        np.random.seed(123)
        sequences = np.memmap(config.data_dir, 
                              shape=(10000,20,self.config.input_shape[0], self.config.input_shape[1], 1),
                              mode='r', dtype=np.float32)
        shuffled_idxs = np.arange(sequences.shape[0])

        # print(('data shape', sequences.shape))

        self.train_sequences = sequences[:config.train_sequences_num]
        self.test_sequences = sequences[config.train_sequences_num:]

    def next_batch(self):
        idx = np.random.choice(self.train_sequences.shape[0], self.config.batch_size)
        current_sequence = self.train_sequences[idx]

        return current_sequence[:, :self.config.truncated_steps + 1]

    def test_batch(self):
        idx = np.random.choice(self.test_sequences.shape[0], self.config.batch_size)
        current_sequence = self.test_sequences[idx]

        return current_sequence[:, :self.config.truncated_steps + 1]
