import unittest

import torch

from common.storage import LirlStorage


class StorageTest(unittest.TestCase):
    def test_meg(self):
        n_val_envs = 2
        storage = LirlStorage(obs_shape=(64, 64, 3),
                              hidden_state_size=2,
                              num_steps=15,
                              num_envs=10,
                              device='cpu',
                              act_shape=(2,))
        generator = storage.fetch_train_generator(mini_batch_size=5, recurrent=False, valid_envs=n_val_envs,
                                                  valid=False)
        for sample in generator:
            elementwise_meg = torch.rand_like(sample[3])
            indices = sample[-1]
            storage.store_meg(elementwise_meg, indices, n_val_envs)
        storage.done_batch = torch.randint(0,2, storage.done_batch.shape)
        storage.full_meg(0.99, n_val_envs)


if __name__ == '__main__':
    unittest.main()
