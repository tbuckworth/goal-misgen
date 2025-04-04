import numpy as np
import torch
from scipy.special import log_softmax

from .misc_util import orthogonal_init
from .model import GRU
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class CategoricalPolicy(nn.Module):
    def __init__(self, 
                 embedder,
                 recurrent,
                 action_size):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """ 
        super(CategoricalPolicy, self).__init__()
        self.T = 1.
        self.embedder = embedder
        self.action_size = action_size
        # small scale weight-initialization in policy enhances the stability        
        self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)

        self.recurrent = recurrent
        if self.recurrent:
            self.gru = GRU(self.embedder.output_dim, self.embedder.output_dim)

    def is_recurrent(self):
        return self.recurrent

    def forward(self, x, hx, masks):
        hidden = self.embedder(x)
        if self.recurrent:
            hidden, hx = self.gru(hidden, hx, masks)
        logits = self.fc_policy(hidden)/self.T
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v, hx

    def value(self, x):
        hidden = self.embedder(x)
        v = self.fc_value(hidden).reshape(-1)
        return v
    
    def forward_with_embedding(self, x):
        hidden = self.embedder(x)
        logits = self.fc_policy(hidden)/self.T
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v, hidden

class ValuePolicyWrapper(nn.Module):
    def __init__(self, policy):
        super(ValuePolicyWrapper, self).__init__()
        self.policy = policy

    def forward(self, x):
        return self.policy.value(x)

class UniformPolicy(nn.Module):
    def __init__(self, action_size, device, input_dims=1):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(UniformPolicy, self).__init__()
        self.action_size = action_size
        self.device = device
        self.recurrent = False
        self.input_dims = input_dims

    def is_recurrent(self):
        return self.recurrent

    def forward(self, x, hx=None, masks=None):
        bs = list(x.shape[:-self.input_dims])
        act_shape = bs + [self.action_size]
        logits = torch.ones(act_shape).to(device=self.device)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        return p, self.value(x), hx

    def value(self, x):
        bs = list(x.shape[:-self.input_dims])
        v_shape = bs
        return torch.zeros(v_shape).to(device=self.device)

    def forward_with_embedding(self, x):
        return self.forward(x)


class PolicyWrapperIRL(nn.Module):
    def __init__(self,policy, device):
        super(PolicyWrapperIRL, self).__init__()
        self.policy = policy
        self.device = device

    def forward(self, obs, states=None, episode_masks=None):
        x = torch.FloatTensor(obs).to(self.device)
        p, v, hx = self.policy(x, None, None)
        act = p.sample().detach().cpu().numpy()
        return act, None


class CraftedPolicy:
    def __init__(self, misgen=False):
        self.embedder = np.zeros((6, 3))
        self.actor = np.zeros((3, 2))
        self.actor[0, 0] = 1.
        self.actor[2, 1] = 1.
        if not misgen:
            self.embedder[0, 0] = 1.
            self.embedder[2, 1] = 1.
            self.embedder[4, 2] = 1.
        else:
            self.embedder[1, 0] = 1.
            self.embedder[3, 1] = 1.
            self.embedder[5, 2] = 1.

    def embed(self, obs):
        return obs @ self.embedder

    def forward(self, obs, embed=True):
        if embed:
            h = self.embed(obs)
        else:
            h = obs
        logits = h @ self.actor
        return log_softmax(logits, axis=-1)

    def act(self, obs, embed=True):
        logits = self.forward(obs, embed=embed)
        p = np.exp(logits)
        return np.random.choice([-1, 1], p=p)

class CraftedTorchPolicy(nn.Module):
    def __init__(self, misgen, action_size, device, input_dims=1):
        super(CraftedTorchPolicy, self).__init__()
        self.recurrent = False
        self.action_size = action_size
        self.device = device
        self.misgen = misgen
        self.embedder = torch.zeros((6, 3)).to(device)
        self.actor = torch.zeros((3, 2)).to(device)
        self.actor[0, 0] = 1.
        self.actor[2, 1] = 1.
        if not self.misgen:
            self.embedder[0, 0] = 1.
            self.embedder[2, 1] = 1.
            self.embedder[4, 2] = 1.
        else:
            self.embedder[1, 0] = 1.
            self.embedder[3, 1] = 1.
            self.embedder[5, 2] = 1.

    def is_recurrent(self):
        return False

    def embed(self, obs):
        return obs @ self.embedder

    def fc_policy(self, hidden):
        return hidden @ self.actor

    def fc_value(self, hidden):
        return torch.ones((*hidden.shape[:-1],1)).to(self.device)

    def forward(self, x, hx, masks):
        hidden = self.embed(x)

        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v, hx

    def value(self, x):
        hidden = self.embed(x)
        v = self.fc_value(hidden).reshape(-1)
        return v

    def forward_with_embedding(self, x):
        hidden = self.embed(x)
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v, hidden
