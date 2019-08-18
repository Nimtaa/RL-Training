import gym
import itertools
import numpy as np
import sys
import tensorflow as tf
import collections
import random
import time
import pandas as pd
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from plotting import plot
from gym import wrappers
from policy import Policy
from value_function import ValueFunction
import scipy.signal
from scaler import Scaler
from datetime import datetime
import os
import argparse
import signal





def init_gym (env_name):
    env = gym.envs.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    return env, obs_dim, act_dim


def run_episode(env, policy, scaler):
    # Run a single episode
    
    obs = env.reset()
    
    observations, actions, rewards, unscaled_obs = [], [], [], []
    
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # Don't scale time-step feature
    offset[-1] = 0.0  # Don't offset time-step feature

    for t in itertools.count():
        env.render()
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # Add time step feature
        unscaled_obs.append(obs)
        obs = scale * (obs - offset) # Center & scale observations
        observations.append(obs)
        action = policy.sample(obs).reshape((1,-1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action,axis = 0)) # Take an action
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # Increment time step feature
        if done:
            return (np.concatenate(observations), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))

    
def run_policy(env, policy, scaler, stats, episodes):
    # Run policy, collect data
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observations, actions, rewards, unscaled_obs = run_episode(env, policy, scaler)
        total_steps += observations.shape[0]
        trajectory = {'observations': observations,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
        stats.episode_rewards[e] = np.mean(rewards)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    print("M_Reward", np.mean([t['rewards'].sum() for t in trajectories]))
    return trajectories

def discount(x, gamma):
    # Calculate discounted forward sum of a sequence
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def insert_disc_sum_rew(trajectories, gamma):
    # Inserts discounted sum of rewards to all time steps of all trajectories

    for trajectory in trajectories:
        if gamma < 0.999:  # Don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew

def insert_value_estimate(trajectories, val_func):
    # Inserts estimated value to all time-steps trajectories
    for t in trajectories:
        obs = t['observations']
        values = val_func.predict(obs)
        t['values'] = values

def insert_GAE(trajectories, gamma, lamb):
    for trajectory in trajectories:
        if gamma < 0.999:  # Don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # TD
        td = rewards + np.append(gamma * values[1:],0) - values
        advantages = discount(td, gamma * lamb)
        trajectory['advantages'] = advantages

def create_train_set(trajectories):
    """
     Create train set from processed trajectories (insert_disc_sum_rew, 
     insert_value_estimate, insert_GAE)
    """
    observations = np.concatenate([t['observations'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observations, actions, advantages, disc_sum_rew

def main(env_name, num_episodes, gamma, lamb, kl_targ, batch_size, hid1_mult, policy_logvar, clipping_range):
    """
    Main training loop
    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lamb: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
        hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
        policy_logvar: natural log of initial policy variance
    """
    env, obs_dim, act_dim = init_gym(env_name)
    obs_dim += 1 
    scaler = Scaler(obs_dim)
    val_func = ValueFunction(obs_dim, hid1_mult)
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar, clipping_range)

    # Keeps track of training information
    EpisodeStats = collections.namedtuple("Stats",["episode_lengths", "episode_rewards"])

    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    


    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, stats ,episodes=5)

    episode = 0
    while episode < num_episodes :
        print("Episode Num:", episode)
        trajectories = run_policy(env, policy, scaler, stats ,episodes=batch_size)
        episode += len(trajectories)
        insert_value_estimate (trajectories, val_func) # Insert estimated values to episodes
        insert_disc_sum_rew(trajectories, gamma) # Calculate discounter sum of rewards
        insert_GAE(trajectories, gamma, lamb) # Calculate advantages

        # Concatenate all episodes into a single array 
        observations, actions, advantages, disc_sum_rew = create_train_set(trajectories)

        policy.update(observations, actions, advantages) # Policy update
        val_func.fit(observations, disc_sum_rew) # Value Function update
        
    policy.close_sess()
    val_func.close_sess()
    # TODO
    # Plotting, plot method from plotting.py
    plot(stats)
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=1000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lamb', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('-m', '--hid1_mult', type=int,
                        help='Size of first hidden layer for value and policy NNs'
                             '(integer multiplier of observation dimension)',
                        default=10)
    parser.add_argument('-v', '--policy_logvar', type=float,
                        help='Initial policy log-variance (natural log of variance)',
                        default=-1.0)
    parser.add_argument('-c', '--clipping_range',
                        nargs=2, type=float,
                        help='Use clipping range objective in PPO instead of KL divergence penalty',
                        default=None)

    args = parser.parse_args()
    main(**vars(args))






