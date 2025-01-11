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


def make_env(env_id, seed, idx, capture_video, run_name):

    if capture_video and idx == 0:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env, f"logs/videos/dqn_atari/{run_name}", disable_logger=True
        )
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    env = NoopResetEnv(env, noop_max=30)
    # env = MaxAndSkipEnv(env, skip=4)

    env = EpisodicLifeEnv(env)

    # if "FIRE" in env.unwrapped.get_action_meanings():
    #     env = FireResetEnv(env)

    # env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)

    env.action_space.seed(seed)
    return env


if __name__ == "__main__":
    env = make_env("ALE/MontezumaRevenge-v5", 123, 0, True, "tqweqw")

    train_dqn(env)
