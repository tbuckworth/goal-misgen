import torch

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
        logits = self.fc_policy(hidden)
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
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v, hidden

class UniformPolicy(nn.Module):
    def __init__(self, action_size, device):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(UniformPolicy, self).__init__()
        self.action_size = action_size
        self.device = device
        self.recurrent = False

    def is_recurrent(self):
        return self.recurrent

    def forward(self, x, hx=None, masks=None):
        bs = list(x.shape[:-1])
        act_shape = bs + [self.action_size]
        logits = torch.ones(act_shape).to(device=self.device)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        return p, self.value(x), hx

    def value(self, x):
        bs = list(x.shape[:-1])
        v_shape = bs + [1]
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

