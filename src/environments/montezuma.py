import ale_py
import gymnasium as gym
import numpy as np

import os
from .utils import load_img_np, match_tp, rgb_to_grayscale

from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv
from collections import deque
import cv2


class SkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    # taking max of frames will screwed the object detection
    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip

    def step(self, action: int):
        total_reward = 0.0
        terminated = truncated = False

        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            final_obs = obs

            total_reward += float(reward)
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        # max_frame = self._obs_buffer.max(axis=0)

        return final_obs, total_reward, terminated, truncated, info


def find_character(img):
    mask = (img[:, :] == 181) | (img[:, :] == 99)

    # remove the UI area
    mask[:35, :] = False

    t = np.where(mask)

    if len(t[0]) == 0:
        return None

    x, y = np.median(t[0]).astype(int), np.median(t[1]).astype(int)

    return np.array([x, y])


class MontezumaObjectDetectionWrapper(gym.Wrapper):
    def __init__(self, env, templates_dir="templates/montezuma"):
        super().__init__(env)

        assert env.observation_space.shape[1:] == (
            210,
            160,
        ) or env.observation_space.shape == (210, 160)

        self.templates = {
            "door": load_img_np(os.path.join(templates_dir, "door.png")),
            "key": load_img_np(os.path.join(templates_dir, "key.png")),
            "ladder": load_img_np(os.path.join(templates_dir, "ladder.png")),
        }

        self.non_ui_x_range = [48, 200]

        self.cached_char_pos = [84, 80]

        self.cached_door_pos = []
        self.cached_key_pos = []
        self.cached_ladder_pos = []

        # for the goals in hdqn
        self.max_objs = {"char": 1, "key": 1, "ladder": 3, "door": 2}

        self.hit_box = {
            "char": (6, 6),  # 10 in row and columns
            "key": (8, 4),
            "ladder": (8, 4),
            "door": (17, 10),
        }

    def _detect(self, frame):
        # frame (210, 160) grayscale
        char_pos = find_character(frame)

        if char_pos is None:
            char_pos = self.cached_char_pos
        else:
            self.cached_char_pos = char_pos

        if len(self.cached_door_pos) != 2:
            self.cached_door_pos = match_tp(
                frame, self.templates["door"], x_range=self.non_ui_x_range
            )

        if len(self.cached_key_pos) != 1:
            self.cached_key_pos = match_tp(
                frame, self.templates["key"], x_range=self.non_ui_x_range
            )

        if len(self.cached_ladder_pos) != 3:
            self.cached_ladder_pos = match_tp(
                frame, self.templates["ladder"], x_range=self.non_ui_x_range
            )

    def _modify_info(self, info):
        info["char_pos"] = self.cached_char_pos
        info["ladder_pos"] = self.cached_ladder_pos
        info["key_pos"] = self.cached_key_pos
        info["door_pos"] = self.cached_door_pos

        goal_list = [(self.cached_char_pos, True, self.hit_box["char"])]

        for k, v in self.max_objs.items():
            if k == "char":
                continue

            fk = f"{k}_pos"

            for i in range(v):
                # this wil ignore objects after the
                # maximum number of object

                if i < len(info[fk]):
                    goal_list.append((info[fk][i], True, self.hit_box[k]))
                else:
                    goal_list.append(((0, 0), False, self.hit_box[k]))

        info["goal_list"] = goal_list
        return info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if len(obs.shape) == 3:
            # detect the last frame
            self._detect(obs[-1])
        else:
            len(obs.shape) == 2
            self._detect(obs)

        info = self._modify_info(info)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)

        self.cached_door_pos = []
        self.cached_key_pos = []
        self.cached_ladder_pos = []
        self.cached_char_pos = [84, 80]

        if len(obs.shape) == 3:
            # detect the last frame
            self._detect(obs[-1])
        else:
            len(obs.shape) == 2
            self._detect(obs)

        info = self._modify_info(info)

        return obs, info


def calc_dist(p1, p2):
    x, y = p1
    x1, y1 = p2
    return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)


def calc_dist1(p1, p2):
    x, y = p1
    x1, y1 = p2
    return max(np.abs(x - x1), np.abs(y - y1))


def calc_dist2(p1, p2):
    x, y = p1
    x1, y1 = p2
    return np.abs(x - x1), np.abs(y - y1)


class MontezumaIntrinsicRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.max_objs = {"char": 1, "key": 1, "ladder": 3, "door": 2}

    def _set_intrinsic_reward(self, info):
        # give reward of 1 if the character touch the goal
        # or the goal disappear
        # which can be caused by the character interact with it
        # check with all goals at the same time for simplicity
        intrinsic_rewards = []
        dones = []

        for obj_pos, avail, (mdx, mdy) in info["goal_list"]:
            # dist = calc_dist1(info["char_pos"], obj_pos)

            dx, dy = calc_dist2(info["char_pos"], obj_pos)

            done = False
            reward = 0
            if dx <= mdx + 1 and dy <= mdy + 1:
                done = True
                # more reward for being closer
                reward = 1
                # reward = 1 / (1 + dist)

            if not avail:
                done = True
                reward = 1

            # reward = dist

            intrinsic_rewards.append(reward)
            dones.append(done)

        info["intrinsic_rewards"] = intrinsic_rewards
        info["intrinsic_dones"] = dones

        return info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info = self._set_intrinsic_reward(info)

        if terminated and reward == 0:
            info["intrinsic_rewards"] = [0 for _ in info["intrinsic_rewards"]]
            info["intrinsic_dones"] = [True for _ in info["intrinsic_dones"]]

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)

        info = self._set_intrinsic_reward(info)

        return obs, info


class FinWrapper(gym.Wrapper):
    # For hDQN
    OBSERVATION_SPACE_META = gym.spaces.Box(
        low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
    )
    # add a single bit map of goal
    OBSERVATION_SPACE_CONTROLLER = gym.spaces.Box(
        low=0, high=255, shape=(5, 84, 84), dtype=np.uint8
    )

    ACTION_SPACE_META = gym.spaces.Discrete(7)
    # also equal number of subgoals
    # TODO: find better way to encode this
    # 2 doors, 1 key, 3 ladders

    def __init__(self, env):
        super().__init__(env)

        self.ori_h, self.ori_w = env.observation_space.shape[-2:]

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )

    def _get_goal_mask(self, info, goal_idx):
        mask = np.zeros((self.ori_h, self.ori_w), dtype=np.uint8)

        (x, y), avail, (mdx, mdy) = info["goal_list"][goal_idx]

        if not avail:
            return cv2.resize(mask, (84, 84))

        # for simplicity use a 5x5 square to represent the object
        # dx = 5
        # dy = 5
        dx = mdx
        dy = mdy
        x1, x2 = max(x - dx, 0), min(x + dx, mask.shape[0])
        y1, y2 = max(y - dy, 0), min(y + dy, mask.shape[1])

        mask[x1:x2, y1:y2] = 255

        mask = cv2.resize(mask, (84, 84))

        return mask

        # return self._get_observation(np.vstack([self.cached_obs, mask[None, :]]))

    def encode_goal(self, obs, info, goal_idx):
        mask = self._get_goal_mask(info, goal_idx)

        return np.vstack([obs, mask[None, :]])

    def _resize_observation(self, obs):
        obs = cv2.resize(obs.transpose((1, 2, 0)), (84, 84)).transpose((2, 0, 1))
        return obs

    def reset(self):
        obs, info = self.env.reset()

        return self._resize_observation(obs), info

    def step(self, action):
        obs, reward, done, truncation, info = self.env.step(action)

        return self._resize_observation(obs), reward, done, truncation, info


def CreateMontezumaRevengeEnv(track_obj=False, record_vid_path=None):
    env = gym.make("ALE/MontezumaRevenge-v5", render_mode="rgb_array")

    if record_vid_path:
        env = gym.wrappers.RecordVideo(env, record_vid_path)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = SkipEnv(env, skip=4)

    env = EpisodicLifeEnv(env)

    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)

    if track_obj:
        env = MontezumaObjectDetectionWrapper(env)
        env = MontezumaIntrinsicRewardWrapper(env)

    env = FinWrapper(env)
    return env
