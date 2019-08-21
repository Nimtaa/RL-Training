import itertools
import matplotlib
from matplotlib import pyplot as plt
import time
import pandas as pd
import numpy as np
matplotlib.style.use('ggplot')

def plot(stats):
    print(stats)
    smoothing_window = 10
    episode_lengths = stats.episode_lengths / 20
    episode_rewards = stats.episode_rewards
    
    # Plot the episode reward over time
    fig1 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.show(fig1)



