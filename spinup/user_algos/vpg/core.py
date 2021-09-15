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


class MLPValueEstimator(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        # Need to call nn.Module.__init__() first
        super().__init__()
        self.network = mlp([obs_dim] + hidden_sizes + [1], activation=activation)

    def forward(self, obs):
        return torch.squeeze(self.network(obs), -1)


class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, pi_hidden_sizes, v_hidden_sizes, activation):
        super().__init__()
        self.pi = MLPGaussianActor(obs_dim, act_dim, pi_hidden_sizes, activation)
        self.v = MLPValueEstimator(obs_dim, v_hidden_sizes, activation)

    def step(self, obs):
        # We need to calculate values and actions in order to compute the advantage function, but we don't want to
        # backpropagate those calculations
        with torch.no_grad():
            pi, _ = self.pi(obs)
            action = pi.sample()
            value = self.v(obs)
            log_p = pi.log_prob(action)

        # Things work better when they are returned as a numpy array instead of a pytorch tensor
        # Specifically, the gym environment doesn't like pytorch tensors as input
        return action.numpy(), value.numpy(), log_p.numpy()





if __name__ == '__main__':
    obs_dim = 5
    hidden_sizes = [64,64]
    activation = nn.Tanh
    v = MLPValueEstimator(obs_dim, hidden_sizes, activation)

    batch_size = 10
    eps_len = 100
    obs = torch.rand([batch_size, eps_len, obs_dim])
    values = v(obs)

    obs = torch.rand([batch_size, eps_len, obs_dim])
    values = v(obs)