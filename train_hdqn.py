from src.environments import CreateMontezumaRevengeEnv
from src.trainer import train_dqn, train_Hdqn
from src.agents import DQN_Agent, HDQN_Agent

import ale_py
import gymnasium as gym
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from tqdm.auto import tqdm

import time
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

if __name__ == "__main__":
    env = CreateMontezumaRevengeEnv(track_obj=True, record_vid_path="logs/vids")
    # env = CreateMontezumaRevengeEnv(track_obj=True)

    # train_Hdqn(env, learning_starts=20000, learning_rate=2e-5, exploration_fraction=0.1)
    train_Hdqn(env)
    # train_dqn(env)
