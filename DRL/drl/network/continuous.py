import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drl.utils import tensor

class DummyActor(nn.Module):
    def __init__(self, a_dim):
        super().__init__()
        self.action_dim = a_dim

    def forward(self, s):
        act = np.random.rand(s.shape[0], self.action_dim)
        act = tensor(2 * act - 1.)
        return act


class ActorDeter(nn.Module):
    def __init__(self, layer_num, s_dim, a_dim, hidden_dim=128):
        super().__init__()
        self.model = [
            nn.Linear(s_dim, hidden_dim),
            nn.ReLU()]
        for _ in range(layer_num - 2):
            self.model += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.model += [nn.Linear(hidden_dim, a_dim)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, **kwargs):
        s = tensor(s)
        s = s.view(s.shape[0], -1)
        logits = self.model(s)
        logits = torch.tanh(logits)
        return logits


class ActorProb(nn.Module):
    def __init__(self, layer_num, s_dim, a_dim, hidden_dim=256):
        super().__init__()
        self.model = [
            nn.Linear(s_dim, hidden_dim),
            nn.ReLU()]
        for _ in range(layer_num - 2):
            self.model += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(hidden_dim, a_dim)
        self.sigma = nn.Linear(hidden_dim, a_dim)

    def forward(self, s, **kwargs):
        s = tensor(s)
        s = s.view(s.shape[0], -1)
        logits = self.model(s)
        mu = torch.tanh(self.mu(logits))
        sigma = torch.exp(self.sigma(logits))
        return (mu, sigma)


class ActorProb2(nn.Module):
    def __init__(self, layer_num, s_dim, a_dim, hidden_dim=256):
        super().__init__()
        self.model = [
            nn.Linear(s_dim, hidden_dim),
            nn.ReLU()]
        for _ in range(layer_num - 2):
            self.model += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(hidden_dim, a_dim)
        self.sigma = nn.Parameter(-0.5*torch.ones([1, a_dim]))

    def forward(self, s, **kwargs):
        s = tensor(s)
        s = s.view(s.shape[0], -1)
        logits = self.model(s)
        mu = torch.tanh(self.mu(logits))
        sigma = F.softplus(self.sigma).expand(mu.size(0), -1)
        return (mu, sigma)


class Critic(nn.Module):
    def __init__(self, layer_num, s_dim, a_dim=0, hidden_dim=256):
        super().__init__()
        self.model = [
            nn.Linear(s_dim + a_dim, hidden_dim),
            nn.Tanh()]
        for i in range(layer_num - 2):
            self.model += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.model += [nn.Linear(hidden_dim, 1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, a=None):
        s = tensor(s)
        if a is not None:
            a = tensor(a)
        s = s.view(s.shape[0], -1)
        if a is None:
            logits = self.model(s)
        else:
            a = a.view(a.shape[0], -1)
            logits = self.model(torch.cat([s, a], dim=1))
        return logits
