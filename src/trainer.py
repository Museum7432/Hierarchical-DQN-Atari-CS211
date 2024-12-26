import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

import lightning as L

from lightning.fabric.loggers import TensorBoardLogger

from lightning.fabric import Fabric
import os
import random

from .agents import DQN_Agent, HDQN_Agent
from tqdm.auto import tqdm


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def epsilon_greedy_action(agent, epsilon, obs, ACTION_SPACE):
    if random.random() < epsilon:
        action = ACTION_SPACE.sample()
    else:
        action = (
            agent.get_greedy_action(torch.tensor(obs[None, :], device=agent.device))
            .cpu()
            .numpy()[0]
        )
    return action


def train_dqn(
    # agent:DQN_Agent,
    env,
    learning_rate: float = 1e-4,
    batch_size: int = 64,
    total_timesteps: int = 10000000,
    learning_starts: int = 80000,
    train_frequency=64,
    # for replay buffer
    replay_buffer_size: int = 600000,
    target_network_frequency: int = 128,
    # exploration
    start_e: float = 1,
    end_e: float = 0.01,
    exploration_fraction: float = 0.10,
    gamma=0.99,
    tau=0.1,
):
    logger = TensorBoardLogger(root_dir=os.path.join("logs"))

    fabric = Fabric(loggers=logger)

    device = fabric.device
    fabric.seed_everything(69)

    rb = ReplayBuffer(
        replay_buffer_size,
        env.observation_space,
        env.action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    agent = DQN_Agent(
        obs_dims=env.observation_space.shape, n_actions=env.action_space.n
    )
    optimizer = agent.configure_optimizers(lr=learning_rate)
    agent, optimizer = fabric.setup(agent, optimizer)

    obs, _ = env.reset()

    pbar = tqdm(total=total_timesteps)
    global_step = 0

    while global_step < total_timesteps:
        # exploration
        # TODO: use lr_scheduler for this
        epsilon = linear_schedule(
            start_e, end_e, exploration_fraction * total_timesteps, global_step
        )
        action = epsilon_greedy_action(agent, epsilon, obs, env.action_space)

        next_obs, reward, done, truncation, info = env.step(action)

        # ############
        goal_reached = info["intrinsic_dones"][4]
        intrinsic_reward = info["intrinsic_rewards"][4]
        reward = max(intrinsic_reward, reward)
        done = goal_reached or done
        # ############

        global_step += 1
        pbar.update(1)

        if "episode" in info:
            fabric.log("charts/episodic_return", info["episode"]["r"], global_step)
            fabric.log("charts/episodic_length", info["episode"]["l"], global_step)

        rb.add(obs, next_obs, action, reward, done, info)

        obs = next_obs

        if done or truncation:
            obs, _ = env.reset()

        if global_step <= learning_starts:
            continue

        if global_step % train_frequency != 0:
            continue

        batch = rb.sample(batch_size)

        loss, log = agent.training_step(batch)

        if global_step % 100 == 0:
            fabric.log("losses/td_loss", loss.item(), global_step)
            fabric.log("losses/q_values", log["old_val"].mean().item(), global_step)

        optimizer.zero_grad()
        fabric.backward(loss)
        optimizer.step()

        # update target network
        if global_step % target_network_frequency == 0:
            agent.update_target_network(tau=tau)

    return agent


def epsilon_greedy_action_with_mask(agent, epsilon, obs, ACTION_SPACE, dones):
    action = 0
    if random.random() < epsilon:
        # pick something that isnt done
        while dones[action]:
            action = ACTION_SPACE.sample()
    else:

        with torch.no_grad():
            Qvalue = (
                agent.get_value(torch.tensor(obs[None, :], device=agent.device))
                .cpu()
                .numpy()[0]
            )

        Qvalue[dones] = float("-Inf")

        action = np.argmax(Qvalue)

    return action


def update_agent(fabric, optimizer, agent, replay, batch_size=32):
    batch = replay.sample(batch_size)
    loss, log = agent.training_step(batch)

    optimizer.zero_grad()
    fabric.backward(loss)
    # fabric.clip_gradients(agent, optimizer, clip_val=0.5)
    optimizer.step()

    return loss.item(), log["old_val"].mean().item()


def get_eps(
    ep_endt=500000,
    ep_start=1,
    ep_end=0.1,
    learn_start=80000,
    numSteps=500000,
):
    return ep_end + max(
        0,
        (ep_start - ep_end) * (ep_endt - max(0, numSteps - learn_start)) / ep_endt,
    )


def train_Hdqn(
    env,
    batch_size: int = 256,
    total_timesteps: int = 10000000,
    # for cotnroller
    learning_starts: int = 50000,
    replay_buffer_size: int = 600000,
    learning_rate: float = 2.5e-4,
    # for meta cotnroller
    learning_starts_meta: int = 1000,
    learning_rate_meta: float = 5e-4,
    max_step_per_goal: int = 500,
    #
    train_frequency=8,
    target_network_frequency: int = 16,
    # exploration
    start_e: float = 1,
    end_e: float = 0.1,
    exploration_fraction: float = 0.10,
    gamma=0.99,
    tau=1e-3,
):
    logger = TensorBoardLogger(root_dir=os.path.join("logs"))

    fabric = Fabric(loggers=logger)

    device = fabric.device
    fabric.seed_everything(69)

    rb_meta = ReplayBuffer(
        replay_buffer_size,
        env.OBSERVATION_SPACE_META,
        env.ACTION_SPACE_META,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    rb_ctrl = ReplayBuffer(
        replay_buffer_size,
        env.OBSERVATION_SPACE_CONTROLLER,
        env.action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    agent = HDQN_Agent(
        meta_obs_dims=env.OBSERVATION_SPACE_META.shape,
        n_subgoals=env.ACTION_SPACE_META.n,
        obs_dims=env.OBSERVATION_SPACE_CONTROLLER.shape,
        n_actions=env.action_space.n,
    )

    meta_optimizer, ctrl_optimizer = agent.configure_optimizers(lr=learning_rate)
    agent, meta_optimizer, ctrl_optimizer = fabric.setup(
        agent, meta_optimizer, ctrl_optimizer
    )

    obs, info = env.reset()

    pbar = tqdm(total=total_timesteps)
    global_step = 0

    # for logging
    n_goals_in_ep = 0

    while global_step < total_timesteps:

        # select a goal
        # epsilon = linear_schedule(
        #     start_e, end_e, exploration_fraction * total_timesteps, global_step
        # )

        epsilon = get_eps(
            ep_endt=replay_buffer_size,
            ep_start=start_e,
            ep_end=end_e,
            learn_start=learning_starts_meta,
            numSteps=global_step,
        )

        # info["intrinsic_dones"][2] = True

        goal = epsilon_greedy_action_with_mask(
            agent.meta_ctrl,
            epsilon,
            # 1.0,
            obs,
            env.ACTION_SPACE_META,
            info["intrinsic_dones"],
        )
        F = 0
        obs0 = obs.copy()

        # for logging
        total_intrinsic_reward = 0
        intrinsic_reward_per_goal = [0 for _ in range(env.ACTION_SPACE_META.n)]

        goal_step = 0

        goal_reached = False
        done = False
        while not done and not goal_reached and goal_step < max_step_per_goal:

            ctrl_obs = env.encode_goal(obs, info, goal)

            # epsilon_ctrl = linear_schedule(
            #     start_e, end_e, exploration_fraction * total_timesteps, global_step
            # )

            epsilon_ctrl = get_eps(
                ep_endt=replay_buffer_size,
                ep_start=start_e,
                ep_end=end_e,
                learn_start=learning_starts,
                numSteps=global_step,
            )

            # copied from the original code
            # for the key and the ladder under it
            # if goal == 1 or goal == 3:
            #     epsilon_ctrl = 0.1

            action = epsilon_greedy_action(
                agent.ctrl, epsilon_ctrl, ctrl_obs, env.action_space
            )

            next_obs, extrinsic_reward, done, truncation, info = env.step(action)
            global_step += 1
            goal_step += 1
            pbar.update(1)

            if "episode" in info:
                fabric.log("charts/episodic_return", info["episode"]["r"], global_step)
                fabric.log("charts/episodic_length", info["episode"]["l"], global_step)

            goal_reached = info["intrinsic_dones"][goal]
            intrinsic_reward = info["intrinsic_rewards"][goal]

            F += extrinsic_reward
            total_intrinsic_reward += intrinsic_reward
            intrinsic_reward_per_goal[goal] += intrinsic_reward

            next_ctrl_obs = env.encode_goal(next_obs, info, goal)

            rb_ctrl.add(
                ctrl_obs,
                next_ctrl_obs,
                action,
                intrinsic_reward,
                # a game termination is the same as a goal being reached
                done or goal_reached,
                info,
            )

            obs = next_obs

            # update controller
            if rb_ctrl.size() > learning_starts:
                if global_step % train_frequency == 0:
                    loss, prev_Qvalue = update_agent(
                        fabric, ctrl_optimizer, agent.ctrl, rb_ctrl, batch_size
                    )

                    if global_step % 1000 == 0:
                        fabric.log("losses/td_loss", loss, global_step)
                        fabric.log("losses/q_values", prev_Qvalue, global_step)

                if global_step % target_network_frequency:
                    agent.ctrl.update_target_network(tau=tau)

            # TODO: update the meta controller here
            if rb_meta.size() > learning_starts_meta:
                if global_step % (train_frequency * 10) == 0:
                    loss, prev_Qvalue = update_agent(
                        fabric, meta_optimizer, agent.meta_ctrl, rb_meta, batch_size
                    )

                    # if global_step % 1000 == 0:
                    #     fabric.log("losses/td_loss", loss, global_step)
                    #     fabric.log("losses/q_values", prev_Qvalue, global_step)

                if global_step % (target_network_frequency * 10):
                    agent.meta_ctrl.update_target_network(tau=tau)

            if global_step % 200000 == 0:
                state = {"model": agent}
                fabric.save("logs/state.ckpt", state)

        rb_meta.add(obs0, next_obs, goal, F, done, info)

        n_goals_in_ep += 1

        fabric.log("rewards/intrinsic", total_intrinsic_reward, global_step)
        fabric.log(
            f"rewards/intrinsic_{goal}", intrinsic_reward_per_goal[goal], global_step
        )
        fabric.log("rewards/extrinsic", F, global_step)
        fabric.log("epsilon/meta", epsilon, global_step)
        fabric.log("epsilon/ctrl", epsilon_ctrl, global_step)
        
        fabric.log("goals/step_per_goal", goal_step, global_step)

        if done or truncation:
            obs, info = env.reset()

            fabric.log("goals/n_goals_in_ep", n_goals_in_ep, global_step)
            n_goals_in_ep = 0
        

        

    
