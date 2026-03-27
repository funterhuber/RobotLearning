"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""


    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 4,
        chunk_size: int = 16,
        hidden_dim: int = 512,
        num_layers: int = 5,
        activation: str = "Gelu",
        dropout: float = 0.0
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        act_map = {
            "ReLu": nn.ReLU,
            "SwiGLU": nn.SiLU,
            "Gelu": nn.GELU,
        }
        act_cls = act_map[activation]

        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            act_cls(),
            nn.Dropout(p=dropout),
        )

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_cls(),
                nn.Dropout(p=dropout),
            )
            for _ in range(num_layers)
        ])

        # Output head: hidden_dim -> flat action chunk
        self.output_head = nn.Linear(hidden_dim, chunk_size * action_dim)

    def forward(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        x = self.input_proj(state)
        for block in self.res_blocks:
            x = x + block(x)
        x = self.output_head(x)
        return x.reshape(state.shape[0], self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor
    ) -> torch.Tensor:
        pred_actions = self.forward(state)
        return torch.nn.functional.mse_loss(pred_actions, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        actions = self.forward(state)

        return actions


# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene.

    Expects state layout (dim=19):
        ee_xyz(3) + gripper(1) + red_cube_pos(3) + green_cube_pos(3) +
        blue_cube_pos(3) + goal_onehot(3) + goal_pos(3)

    Internally selects the target cube based on goal_onehot, reducing
    the effective input to 10 dims: ee_xyz(3) + gripper(1) + target_cube(3) + goal_pos(3).
    """

    _REDUCED_DIM = 10

    def __init__(
        self,
        state_dim: int = 19,
        action_dim: int = 4,
        chunk_size: int = 16,
        hidden_dim: int = 512,
        activation: str = "Gelu",
        num_layers: int = 5,
        dropout: float = 0.0
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        act_map = {
            "ReLu": nn.ReLU,
            "SwiGLU": nn.SiLU,
            "Gelu": nn.GELU,
        }
        act_cls = act_map[activation]

        self.input_proj = nn.Sequential(
            nn.Linear(self._REDUCED_DIM, hidden_dim),
            act_cls(),
            nn.Dropout(p=dropout),
        )

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_cls(),
                nn.Dropout(p=dropout),
            )
            for _ in range(num_layers)
        ])

        self.output_head = nn.Linear(hidden_dim, chunk_size * action_dim)

    def _select_target_cube(self, state: torch.Tensor) -> torch.Tensor:
        """Reduce 19-dim state to 10-dim by selecting the target cube."""
        ee_xyz = state[:, 0:3]
        gripper = state[:, 3:4]
        red = state[:, 4:7]
        green = state[:, 7:10]
        blue = state[:, 10:13]
        goal_onehot = state[:, 13:16]
        goal_pos = state[:, 16:19]

        # Select target cube: goal_onehot acts as a soft selector
        target_cube = (goal_onehot[:, 0:1] * red +
                       goal_onehot[:, 1:2] * green +
                       goal_onehot[:, 2:3] * blue)

        return torch.cat([ee_xyz, gripper, target_cube, goal_pos], dim=1)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor
    ) -> torch.Tensor:
        pred_action = self.forward(state)
        return torch.nn.functional.mse_loss(pred_action, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self.forward(state)

    def forward(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        x = self._select_target_cube(state)
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = x + block(x)
        x = self.output_head(x)
        return x.reshape(state.shape[0], self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    chunk_size: int,
    hidden_dim: int = 512,
    d_model: int | None = None,
    num_layers: int = 5,
    activation: str = "Gelu",
    dropout: float = 0.0,
    state_dim: int,
    action_dim: int,
    **kwargs,
) -> BasePolicy:
    if d_model is not None:
        hidden_dim = d_model
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
            hidden_dim= hidden_dim,
            num_layers= num_layers,
            activation=activation,
            dropout=dropout,
            chunk_size=chunk_size
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            chunk_size=chunk_size
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
