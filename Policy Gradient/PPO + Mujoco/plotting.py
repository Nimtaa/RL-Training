import itertools
import matplotlib
from matplotlib import pyplot as plt
import time
import pandas as pd
import numpy as np
matplotlib.style.use('ggplot')



def plot(stats):
    # smoothing_window = 5
    print("len :", len(stats.episode_lengths))
    temp = int (len(stats.episode_lengths)/20)
    print("temp",temp)
    print("epl befor", stats.episode_lengths)
    episode_lengths = np.resize(stats.episode_lengths, (temp,))
    print(episode_lengths)
    episode_rewards = np.resize(stats.episode_rewards, (temp,))
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    # rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward over Time")
    plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(episode_lengths), np.arange(len(episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.show(fig3)



