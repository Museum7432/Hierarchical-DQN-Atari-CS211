import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L


def calculate_flatten_size(h, w, fin_c):
    # (h - k)//s + 1

    # stride: 4, kernel: 8
    h = (h - 8) // 4 + 1
    w = (w - 8) // 4 + 1

    # stride: 2, kernel: 4
    h = (h - 4) // 2 + 1
    w = (w - 4) // 2 + 1

    # stride: 1, kernel: 3
    h = h - 2
    w = w - 2

    return h * w * fin_c


class QNetwork(nn.Module):
    def __init__(self, h=210, w=160, in_c=3, n_actions=18):
        super().__init__()

        fl_size = calculate_flatten_size(h, w, 64)

        self.network = nn.Sequential(
            nn.Conv2d(in_c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fl_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class DQN_Agent(L.LightningModule):
    # TODO: do not save the target network in the checkpoint

    def __init__(self, obs_dims=(4, 84, 84), n_actions=18, use_target_net=True):
        super().__init__()
        self.save_hyperparameters("obs_dims", "n_actions")

        # TODO: use DDQN agent
        self.use_target_net = use_target_net

        # only work with atari games for now
        assert len(obs_dims) == 3
        c, h, w = obs_dims

        self.q_network = QNetwork(h, w, c, n_actions=n_actions)

        if use_target_net:
            self.target_network = QNetwork(h, w, c, n_actions=n_actions)
            self.update_target_network(copy=True)
            # Freeze the target network parameters
            # for param in self.target_network.parameters():
            #     param.requires_grad = False

    def forward(self, obs):
        return self.q_network(obs)

    @torch.no_grad()
    def update_target_network(self, tau=0.1, copy=False):
        if copy:
            self.target_network.load_state_dict(self.q_network.state_dict())
            return

        for target_network_param, q_network_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_network_param.data.copy_(
                tau * q_network_param.data + (1.0 - tau) * target_network_param.data
            )

    @torch.no_grad()
    def get_greedy_action(self, obs):
        q_values = self.q_network(obs)

        return torch.argmax(q_values, dim=1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_network(obs)

    def training_step(self, batch: dict[str, torch.Tensor], gamma=0.99):
        with torch.no_grad():
            if self.use_target_net:
                target_max, _ = self.target_network(batch.next_observations).max(dim=1)
            else:
                target_max, _ = self.q_network(batch.next_observations).max(dim=1)

            td_target = batch.rewards.flatten() + gamma * target_max * (
                1 - batch.dones.flatten()
            )

        old_val = self.get_value(batch.observations).gather(1, batch.actions).squeeze()

        loss = F.mse_loss(td_target, old_val)

        return loss, {"old_val": old_val, "td_target": td_target}

    def configure_optimizers(self, lr: float):
        # learning rate scheduler is managed in the training loop
        return torch.optim.Adam(self.q_network.parameters(), lr=lr)


class HDQN_Agent(L.LightningModule):
    def __init__(
        self,
        # meta cotnroller
        meta_obs_dims=(4, 84, 84),  # concat 4 consecutive frames
        n_subgoals=8,  # .e.g: number of objects in atari game
        # controller
        obs_dims=(5, 84, 84),  # concat 4 consecutive frames
        n_actions: int = 18,
        use_target_net=True,
    ):
        super().__init__()
        self.save_hyperparameters(
            "obs_dims", "n_actions", "meta_obs_dims", "n_subgoals"
        )

        self.meta_ctrl = DQN_Agent(
            obs_dims=meta_obs_dims,
            n_actions=n_subgoals,
            use_target_net=use_target_net,
        )

        self.ctrl = DQN_Agent(
            obs_dims=obs_dims, n_actions=n_actions, use_target_net=use_target_net
        )

    @torch.no_grad()
    def update_target_network(self, tau=0.1, copy=False):
        self.meta_ctrl.update_target_network(tau, copy=copy)
        self.ctrl.update_target_network(tau, copy=copy)

    def configure_optimizers(self, lr: float):
        meta_opt = self.meta_ctrl.configure_optimizers(lr=lr)
        ctrl_opt = self.ctrl.configure_optimizers(lr=lr)
        return meta_opt, ctrl_opt
