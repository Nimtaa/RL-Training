import itertools
import matplotlib
from matplotlib import pyplot as plt
import time
import pandas as pd
matplotlib.style.use('ggplot')


def plot(stats):
    smoothing_window = 100
    episode_lengths = stats[0][0]
    episode_rewards = stats[0][1]
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(episode_lengths), np.arange(len(episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.show(fig3)

    # Plot value_loss and episode number
    fig4 = plt.figure(figsize=(10,5))
    value_loss_smoothed = pd.Series(stats[1]).rolling(25, min_periods=25).mean()
    plt.plot(value_loss_smoothed)
    plt.xlabel("Time Steps")
    plt.ylabel("Value Loss")
    plt.title("Value loss per time step")
    plt.show(fig4)

    # Plot policy_loss and episode number
    fig5 = plt.figure(figsize=(10,5))
    value_loss_smoothed = pd.Series(stats[2]).rolling(25, min_periods=25).mean()
    plt.plot(value_loss_smoothed)
    plt.xlabel("Time Steps")
    plt.ylabel("Policy Loss")
    plt.title("Policy loss per time step")
    plt.show(fig5)


