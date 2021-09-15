import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box


def mlp(sizes, activation, output_activation=nn.Identity):
    num_layers = len(sizes)
    layers = []
    for i in range(num_layers-2):
        layers += [nn.Linear(in_features=sizes[i], out_features=sizes[i+1]), activation()]

    layers += [nn.Linear(sizes[num_layers-2], sizes[num_layers-1]), output_activation()]
    return nn.Sequential(*layers)

class DiagonalGaussianDistribution:

    def __init__(self, mu, log_std):
        self.mu = mu
        self.log_std = log_std
        self.distribution = torch.distributions.multivariate_normal.MultivariateNormal(loc=self.mu,
                            covariance_matrix=torch.diag(torch.exp(self.log_std)**2))

    def sample(self):
        # Variance is equal to std_dev^2
        return self.distribution.sample()

    def log_prob(self, data):
        return self.distribution.log_prob(data)


class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        # Need to call nn.Module.__init__() first
        super().__init__()
        self.mu_net = mlp([obs_dim] + hidden_sizes + [act_dim], activation)
        self.log_std = nn.Parameter(torch.Tensor(act_dim).fill_(-0.5))

    def forward(self, obs, act=None):
        mu = self.mu_net(obs)
        pi = DiagonalGaussianDistribution(mu, self.log_std)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a


class MLPQEstimator(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        # Need to call nn.Module.__init__() first
        super().__init__()
        self.network = mlp([obs_dim + act_dim] + hidden_sizes + [1], activation=activation)

    def forward(self, obs, act):
        # Need to be careful of dimensions here
        input = torch.cat((obs, act), dim=-1)
        return torch.squeeze(self.network(input), -1)


class MLPQEstimator2(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        # Need to call nn.Module.__init__() first
        super().__init__()
        self.network = mlp([obs_dim] + hidden_sizes + [act_dim], activation=activation)

    def forward(self, obs):
        # Need to be careful of dimensions here
        return torch.squeeze(self.network(obs), -1)


class MLPActorCritic2(nn.Module):

    def __init__(self, observation_space, action_space, q_hidden_sizes, activation):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_range = action_space.n

        self.q = MLPQEstimator2(obs_dim, act_range, q_hidden_sizes, activation)
        self.num_acts = act_range
        self.obs_dim = obs_dim

    def step(self, obs):
        with torch.no_grad():
            q_values = self.q(obs)
            max_q, action = torch.max(q_values, dim=-1)

        # Things work better when they are returned as a numpy array instead of a pytorch tensor
        # Specifically, the gym environment doesn't like pytorch tensors as input
        return action.numpy(), max_q.numpy()


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, q_hidden_sizes, activation):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_range = action_space.n

        self.q = MLPQEstimator(obs_dim, 1, q_hidden_sizes, activation)
        self.possible_acts = torch.as_tensor(range(act_range), dtype=torch.float32).view(act_range, 1)
        self.num_acts = act_range
        self.obs_dim = obs_dim

    def step(self, obs):
        if len(obs.shape) == 2:
            # Assume that batch is present
            batch_size = obs.shape[0]
            state_tensor = torch.repeat_interleave(obs, self.num_acts, dim=0).view(batch_size, self.num_acts, self.obs_dim)
            acts_tensor = self.possible_acts.repeat(obs.shape[0], 1, 1)
        elif len(obs.shape) == 1:
            acts_tensor = self.possible_acts
            state_tensor = obs.repeat(self.num_acts, 1)
        else:
            raise ValueError("Unexpected dimensions")

        with torch.no_grad():
            q_values = self.q(state_tensor, acts_tensor)
            # q_values = torch.squeeze(torch.as_tensor(q_values, dtype=torch.float32), -1)
            max_q, action = torch.max(q_values, dim=-1)

        # Things work better when they are returned as a numpy array instead of a pytorch tensor
        # Specifically, the gym environment doesn't like pytorch tensors as input
        return action.numpy(), max_q.numpy()
