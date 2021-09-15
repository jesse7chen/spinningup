from spinup.user_algos.vpg.core import mlp, MLPActorCritic
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from spinup.utils.logx import EpochLogger
import scipy.signal
import time

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using TD-error
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.rtg_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.sp_adv_buf = np.zeros(size, dtype=np.float32)

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        episode_slice = slice(self.path_start_idx, self.ptr)

        value_plus = np.append(self.val_buf[episode_slice], last_val)
        rew_plus = np.append(self.rew_buf[episode_slice], last_val)

        # Calculate advantage using TD(0) error as estimate
        # advantage = self.rew_buf[episode_slice] + \
        #     self.gamma*value_plus[1:] - \
        #     self.val_buf[episode_slice]
        #
        # self.adv_buf[episode_slice] = advantage

        # Calculate GAE-Lambda calculation
        for t in range(self.path_start_idx, self.ptr):
            self.adv_buf[t] = self.gae_advantage(t, last_val)

        # # the next two lines implement GAE-Lambda advantage calculation
        # deltas = rew_plus[:-1] + self.gamma * value_plus[1:] - value_plus[:-1]
        # self.adv_buf[episode_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # Calculate rewards to go
        self.rtg_buf[episode_slice] = self.calculate_rtg(last_val)

        self.path_start_idx = self.ptr

    def reset(self):
        assert self.ptr == self.max_size  # buffer has to be full before you can reset
        self.ptr, self.path_start_idx = 0, 0


    def calculate_rtg(self, last_val):
        episode_rew = np.append(self.rew_buf[self.path_start_idx:self.ptr], last_val)

        n = len(episode_rew)
        rtg = np.zeros_like(episode_rew)
        # Initialize first value
        rtg[n - 1] = episode_rew[n - 1]

        for i in reversed(range(n - 1)):
            rtg[i] = episode_rew[i] + self.gamma * rtg[i + 1]
        return rtg[:-1]

    def td_0_advantage(self, episode_slice, value_plus):
        return self.rew_buf[episode_slice] + \
            self.gamma*value_plus[1:] - \
            self.val_buf[episode_slice]

    def gae_advantage(self, t, last_val):
        advantage = 0
        value_plus = np.append(self.val_buf, last_val)
        for l in range(self.ptr - t):
            advantage += self.td_residual(value_plus, t, l)*(self.gamma*self.lam)**l
        return advantage

    def td_residual(self, value_plus, t, l):
        index = t+l
        return self.gamma*value_plus[index+1] + self.rew_buf[index] - value_plus[index]


def compute_discounted_rtg(x, discounts):
    return scipy.signal.lfilter([1], [1, float(-discounts)], x[::-1], axis=0)[::-1]


def user_compute_rtg(rewards, discount):
    n = len(rewards)
    rtg = np.zeros_like(rewards)
    # Initialize first value
    rtg[n-1] = rewards[n-1]

    for i in reversed(range(n-1)):
        rtg[i] = rewards[i] + discount*rtg[i+1]
    return rtg


def reward_to_go(x, discount):
    pass

    # n = len(rewards)
    # # Our calculation is assuming that terminal state has zero reward
    #
    # rtg = np.zeros_like(rewards)
    # # Initialize first value
    # rtg[n-1] = rewards[n-1]
    #
    # for i in reversed(range(n-1)):
    #     rtg[i] = rewards[i] + rtg[i+1]
    # return rtg

def vpg(env_name='HalfCheetah-v2', seed=0, max_episode_length=1500, epochs=100, batch_size=4000, gamma=0.99,
        logger_kwargs=dict(), save_freq=10):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = gym.make(env_name), gym.make(env_name)
    env.reset()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    ac = MLPActorCritic(obs_dim, act_dim, [64, 64], [64, 64], nn.Tanh)

    v_optimizer = Adam(ac.v.parameters(), lr=1e-3)
    pi_optimizer = Adam(ac.pi.parameters(), lr=1e-3)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    buf = VPGBuffer(obs_dim, act_dim, batch_size, gamma)
    start_time = time.time()

    def compute_policy_loss(observations, actions, advantage):
        _, log_p = ac.pi(observations, actions)

        # Multiple by -1 to do gradient ascent
        return (-1*log_p * advantage).mean()

    def compute_value_loss(observations, rtg):
        # Monte Carlo Value Function Approximation
        # obs should be a tensor of size [batch_size, ep_size, obs_dim]
        # rtg should be a tensor of size [batch_size, ep_size, 1]
        values = ac.v(observations)
        # values should be of size [batch_size, ep_size, 1]
        batch_mse = ((values - rtg)**2).mean()
        return batch_mse

    def update_parameters():
        observations = torch.as_tensor(buf.obs_buf, dtype=torch.float32)
        actions = torch.as_tensor(buf.act_buf, dtype=torch.float32)
        advantage = torch.as_tensor(buf.adv_buf, dtype=torch.float32)
        returns = torch.as_tensor(buf.rtg_buf, dtype=torch.float32)

        pi_optimizer.zero_grad()
        pi_loss = compute_policy_loss(observations, actions, advantage)
        pi_loss.backward()
        pi_optimizer.step()

        v_optimizer.zero_grad()
        v_loss = compute_value_loss(observations, returns)
        v_loss.backward()
        v_optimizer.step()

        buf.reset()

    for epoch in range(epochs):

        v_optimizer.zero_grad()
        pi_optimizer.zero_grad()

        obs = env.reset()
        ep_len = 0
        ep_rew = 0
        batch_rew = []

        for t in range(batch_size):
            # Sample and store actions (gradient descent turned off for step function)
            action, value, log_p = ac.step(torch.as_tensor(obs, dtype=torch.float32))

            new_obs, r, d, info = env.step(action)
            buf.store(obs, action, r, value, log_p)
            logger.store(VVals=value)

            obs = new_obs

            # Break if max episode length is reached, batch size is reached, or if terminal state is reached
            ep_len += 1
            ep_rew += r

            terminal = (d is True)
            end_of_batch = (t == batch_size-1)
            end_of_episode = (ep_len == max_episode_length)

            if terminal or end_of_episode or end_of_batch:
                # Calculate advantage function and rtg for this episode
                if not terminal:
                    # Have to bootstrap what last value would have been to do rtg and advantage calculation
                    _, v, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                    buf.finish_path(last_val=v)
                else:
                    buf.finish_path(last_val=0)
                    logger.store(EpRet=ep_rew, EpLen=ep_len)

                if not end_of_batch:
                    batch_rew.append(ep_rew)

                obs, ep_len, ep_rew = env.reset(), 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        update_parameters()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * batch_size)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='user_vpg')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name + '-' + args.env.lower(), args.seed)

    vpg(env_name=args.env, logger_kwargs=logger_kwargs, epochs=args.epochs)

    # rewards = np.arange(10, dtype=np.float32)
    # user_rtg = user_compute_rtg(rewards, 0.99)
    # comp_rtg = compute_discounted_rtg(rewards, 0.99)
    #
    # if (user_rtg == rtg):
    #     print("Matches")




    # Old code before implementing buffer

    # for e in range(epochs):
    #     batch_obs = []  # for observations
    #     batch_acts = []  # for actions
    #     batch_rews = []  # for measuring episode returns
    #     batch_rtg = []
    #     batch_len = []
    #     batch_ep_rews = []
    #     batch_logp = []
    #
    #     v_optimizer.zero_grad()
    #     pi_optimizer.zero_grad()
    #
    #     for b in range(batch_size):
    #         ep_rews = []
    #         ep_obs = []
    #         ep_acts = []
    #
    #         obs = env.reset()
    #         ep_obs.append(obs.copy())
    #         for t in range(max_episode_length):
    #             # Sample and store actions
    #             pi, _ = ac.pi(torch.as_tensor(obs, dtype=torch.float32))
    #             action = np.asarray(np.clip(pi.sample(), -act_limit, act_limit))
    #
    #             obs, r, d, info = env.step(action)
    #             ep_acts.append(action)
    #             ep_rews.append(r)
    #             ep_obs.append(obs)
    #
    #             if d is True:
    #                 # print("Episode finished after {} timesteps".format(t + 1))
    #                 # Add episode history to batch
    #                 batch_obs.append(torch.as_tensor(ep_obs, dtype=torch.float32))
    #                 batch_rews.append(torch.as_tensor(ep_rews, dtype=torch.float32))
    #                 batch_acts.append(torch.as_tensor(ep_acts, dtype=torch.float32))
    #                 batch_rtg.append(torch.as_tensor(reward_to_go(ep_rews), dtype=torch.float32))
    #                 batch_len.append(t)
    #                 batch_ep_rews.append(sum(ep_rews))
    #                 break
    #
    #     update_parameters()

    # def update_parameters():
    #     print("Avg episode length: {}".format(sum(batch_len)/batch_size))
    #     print("Avg episode reward: {}".format(sum(batch_ep_rews)/batch_size))
    #     batch_pi_loss = 0
    #     batch_v_loss = 0
    #     for i in range(batch_size):
    #         batch_pi_loss += compute_policy_loss(batch_obs[i], batch_acts[i], batch_rews[i])
    #         batch_v_loss += compute_value_loss(batch_obs[i][:-1], batch_rtg[i])
    #
    #     batch_pi_loss /= batch_size
    #     batch_v_loss /= batch_size
    #     batch_pi_loss.backward()
    #     batch_v_loss.backward()
    #
    #     logger.store(LossPi=batch_pi_loss.item())
    #     logger.store(LossV=batch_v_loss.item())
    #
    #     v_optimizer.step()
    #     pi_optimizer.step()

    # def compute_policy_loss(observations, actions, rewards):
    #     # obs should be a tensor of size [batch_size, ep_size+1, obs_dim]
    #     # rtg should be a tensor of size [batch_size, ep_size, act_dim]
    #     _, log_p = ac.pi(observations[:-1], actions)
    #     # log_p should be of size [batch_size, ep_size, 1]
    #     advantage = compute_advantage(observations, rewards)
    #
    #     # Multiple by -1 to do gradient ascent
    #     return (-1*log_p * advantage).mean()
    #
    # def compute_advantage(observations, rewards):
    #     # Use TD-error for advantage estimation for now
    #     # Freeze value estimator so that gradients are not calculated on it
    #     # obs should include terminal state? Or should we use zero for terminal state value
    #     for p in ac.v.parameters():
    #         p.requires_grad = False
    #
    #     values = ac.v(observations)
    #     advantage = rewards + gamma*values[1:] - values[:-1]
    #
    #     for p in ac.v.parameters():
    #         p.requires_grad = True
    #
    #     return advantage
    #
    # def compute_value_loss(observations, rtg):
    #     # Monte Carlo Value Function Approximation
    #     # obs should be a tensor of size [batch_size, ep_size, obs_dim]
    #     # rtg should be a tensor of size [batch_size, ep_size, 1]
    #     values = ac.v(observations)
    #     # values should be of size [batch_size, ep_size, 1]
    #     batch_mse = ((values - rtg)**2).mean()
    #     return batch_mse