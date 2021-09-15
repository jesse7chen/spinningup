from spinup.user_algos.DQN.core import mlp, MLPActorCritic2
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


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class ReplayBuffer:
    """
    A buffer for storing transitions experienced by a DQN agent interacting
    with the environment
    """

    def __init__(self, obs_dim, act_dim, size):
        self.data_type = np.dtype([('o', np.float32, obs_dim), ('a', np.float32, act_dim), ('r', np.float32),
                                   ('o2', np.float32, obs_dim), ('d', np.float32)])

        self.buffer = np.zeros(size, dtype=self.data_type)
        self.rng = np.random.default_rng()

        self.ptr, self.max_size = 0, size
        self.buffer_full = False

    def store(self, obs, act, rew, new_obs, done):
        assert self.ptr < self.max_size
        self.buffer[self.ptr] = (obs, act, rew, new_obs, done)
        self.ptr += 1

        # Buffer has finite space, loop around when it overflows
        if self.ptr == self.max_size:
            self.ptr = 0
            self.buffer_full = True

    def sample_batch(self, batch_size=32):
        # DQN paper uses batch sizes of 32
        if self.buffer_full:
            batch = self.rng.choice(self.buffer, size=batch_size, replace=False)
        else:
            batch = self.rng.choice(self.buffer[:self.ptr], size=batch_size, replace=False)
        return batch



def dqn(env_name='CartPole-v0', seed=0, max_episode_length=1500, epochs=100, steps_per_epoch=4000, gamma=0.99,
        logger_kwargs=dict(), buffer_size=1000000, save_freq=10, update_start=1000, update_freq=50,
        target_update_freq=1000, start_exploring_steps=10000, q_lr=1e-3, minibatch_size=32, min_epsilon=0.1,
        decrease_epsilon_steps=3e5, testing_steps=3e5, num_test_episodes=10):

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = gym.make(env_name), gym.make(env_name)
    env.reset()

    # Check that environment action space is discrete
    assert isinstance(env.action_space, Discrete), "Environment must have discrete action space"

    obs_dim = env.observation_space.shape[0]
    act_dim = 1

    ac = MLPActorCritic2(env.observation_space, env.action_space, [256, 256], nn.ReLU)
    target_ac = MLPActorCritic2(env.observation_space, env.action_space, [256, 256], nn.ReLU)

    q_optimizer = Adam(ac.q.parameters(), lr=1e-3)
    target_q_optimizer = Adam(target_ac.q.parameters(), lr=1e-3)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    var_counts = tuple(count_vars(module) for module in [ac.q])
    logger.log('\nNumber of parameters: \t q: %d\n' % var_counts)

    buf = ReplayBuffer(obs_dim, act_dim, buffer_size)
    start_time = time.time()
    total_steps = 0

    def compute_q_loss(q_values, target):
        return ((q_values-target)**2).mean()

    def compute_huber_loss(q_values, target):
        criterion = nn.SmoothL1Loss()
        assert q_values.shape == target.shape
        return criterion(q_values, target)

    def update_target_network():
        target_ac.q.load_state_dict(ac.q.state_dict())

    def update_parameters():
        # Grab minibatch from replay buffer
        data = buf.sample_batch(minibatch_size)

        # Calculate target q values (assuming greedy policy with respect to target network)
        _, q_target_values = target_ac.step(torch.as_tensor(data['o2'], dtype=torch.float32))
        target_values = data['r'] + gamma*(1-data['d'])*q_target_values
        target_values = torch.as_tensor(target_values, dtype=torch.float32)

        # Calculate expected q value for transitions
        q_optimizer.zero_grad()
        # q_values = ac.q(torch.as_tensor(data['o'], dtype=torch.float32), torch.as_tensor(data['a'], dtype=torch.float32)
        #                 .view(-1, 1))
        q_values = torch.squeeze(ac.q(torch.as_tensor(data['o'], dtype=torch.float32)).
                                 gather(1, torch.as_tensor(data['a'], dtype=torch.long).view(-1, 1)), -1)

        loss = compute_q_loss(q_values, target_values)
        loss.backward()
        q_optimizer.step()

        with torch.no_grad():
            residual_variance = (target_values-q_values).var()/target_values.var()

        logger.store(LossQ=loss.item(), QVals=q_values.detach().numpy(), ResVar=residual_variance)

    def get_epsilon(num_steps):
        if num_steps > decrease_epsilon_steps:
            return min_epsilon
        else:
            return 1-((1-min_epsilon)*num_steps/decrease_epsilon_steps)

    for epoch in range(epochs):
        q_optimizer.zero_grad()

        obs = env.reset()
        ep_len = 0
        ep_rew = 0
        batch_rew = []

        for t in range(steps_per_epoch):
            # Sample and store actions (s
            # action, value, log_p = ac.step(torch.as_tensor(obs, dtype=torch.float32))

            # Randomly sample actions at first
            if total_steps < start_exploring_steps:
                action = env.action_space.sample()
            else:
                # After a certain number of steps, use epsilon-greedy policy
                epsilon = get_epsilon(total_steps - start_exploring_steps)
                logger.store(epsilon=epsilon)
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))

            new_obs, r, d, info = env.step(action)
            buf.store(obs, action, r, new_obs, d)

            obs = new_obs

            ep_len += 1
            total_steps += 1
            ep_rew += r

            terminal = (d is True)
            end_of_episode = (ep_len == max_episode_length)

            if terminal or end_of_episode:
                logger.store(EpRet=ep_rew, EpLen=ep_len)
                obs, ep_len, ep_rew = env.reset(), 0, 0

            # If buffer is full enough, start updating networks
            if total_steps > update_start:
                if total_steps % update_freq == 0:
                    for _ in range(update_freq):
                        update_parameters()
                if total_steps % target_update_freq == 0:
                    update_target_network()


        # Test model at end of epoch
        for i in range(num_test_episodes):
            obs, ep_len, ep_rew, terminal, end_of_episode = env.reset(), 0, 0, False, False
            while terminal is not True and end_of_episode is not True:
                action, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                new_obs, r, d, info = env.step(action)
                buf.store(obs, action, r, new_obs, d)

                obs = new_obs
                ep_len += 1
                total_steps += 1
                ep_rew += r

                terminal = (d is True)
                end_of_episode = (ep_len == max_episode_length)

            logger.store(TestEpRet=ep_rew, TestEpLen=ep_len)



        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TestEpRet', with_min_and_max=True)
        logger.log_tabular('TestEpLen', average_only=True)
        logger.log_tabular('epsilon', get_epsilon(total_steps - start_exploring_steps))
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('QVals', with_min_and_max=True)
        logger.log_tabular('LossQ', average_only=True)
        logger.log_tabular('ResVar', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='dqn')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name + '-' + args.env.lower(), args.seed)

    dqn(env_name=args.env, logger_kwargs=logger_kwargs, epochs=args.epochs)
