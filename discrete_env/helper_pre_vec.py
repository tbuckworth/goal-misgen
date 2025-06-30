import numpy as np


class StartSpace:
    def __init__(self, low, high, np_random, condition=None, discrete_space=False):
        assert len(low) == len(high), "low and high must be the same length"
        self.low = np.array(low)
        self.high = np.array(high)
        self._np_random = np_random
        self.n_inputs = len(low)
        self.condition = condition
        self.discrete_space = discrete_space

    def sample(self, n):
        x = self.sample_floats(n)
        if self.discrete_space:
            return np.round(x).astype(int)
        return x

    def sample_floats(self, n):
        if self.condition is None:
            return self._np_random.uniform(low=self.low, high=self.high, size=(n, self.n_inputs))

        x = self._np_random.uniform(low=self.low, high=self.high, size=(n*10, self.n_inputs))
        x = x[self.condition(x) == False]
        while len(x) < n:
            z = self._np_random.uniform(low=self.low, high=self.high, size=(n * 10, self.n_inputs))
            z = z[self.condition(z) == False]
            x = np.vstack((x,z))
        return x[0:n]


    def set_np_random(self, np_random: np.random.Generator):
        self._np_random = np_random


def override_value(env_args, hyperparameters, suffix, param, default_value):
    env_args[param] = hyperparameters.get(f"{param}{suffix}", default_value)


def assign_env_vars(hyperparameters, is_valid, defaults):
    env_args = {}
    suffix = ""
    i = 0
    if is_valid:
        suffix = "_v"
        i = -1
    for k, v in defaults.items():
        if is_valid and k == "n_envs":
            override_value(env_args, hyperparameters, "", k, v[i])
        override_value(env_args, hyperparameters, suffix, k, v[i])
    extras = {k:v for k,v in hyperparameters.items() if k not in env_args and not k.endswith("_v")}
    env_args.update(extras)
    return env_args

class DictToArgs:
    def __init__(self, input_dict):
        for key in input_dict.keys():
            setattr(self, key, input_dict[key])