%matplotlib inline
import gym
import itertools
import matplotlib
from matplotlib import pyplot as plt
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
matplotlib.style.use('ggplot')


#TODO place it in the main
# Keeps track of training information
EpisodeStats = collections.namedtuple("Stats",["episode_lengths", "episode_rewards"])

stats = EpisodeStats(
    episode_lengths=np.zeros(num_episodes),
    episode_rewards=np.zeros(num_episodes))    

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
        obs = scale * (obse - offset) # Center & scale observations
        observations.append(obs)
        action = policy.sample(obs).reshape((1,-1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action,axis = 0)) # Take an action
        rewards.append(reward)
        step += 1e-3  # Increment time step feature

        if done:
            return (np.concatenate(observations), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))

    
def run_policy(env, policy, scaler, episodes):
    # Run policy, collect data


