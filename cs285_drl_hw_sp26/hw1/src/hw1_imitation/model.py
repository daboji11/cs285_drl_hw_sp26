"""Model definitions for Push-T imitation policies."""

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
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""



class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], out_dim),
        )


    def forward(self, x):
        return self.net(x)


class MSE(nn.Module):

    def __init__(self):
        super().__init__()

    def loss(self, first, second):
        loss = torch.norm((first - second), p=2, dim=(1, 2))** 2
        return loss.mean()


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        self.hidden_dims = hidden_dims
        super().__init__(state_dim, action_dim, chunk_size)
        self.model = MLP(state_dim, action_dim * chunk_size, hidden_dims)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # compute MSE loss 
        pred_chunk = self.sample_actions(state)
        mse = MSE()
        return mse.loss(first=pred_chunk, second=action_chunk)
      

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        pred_chunk = self.model(state)
        pred_chunk = pred_chunk.view(
            state.shape[0],
            self.chunk_size,
            self.action_dim,
        )
        return pred_chunk




        




class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.hidden_dims = hidden_dims
        self.model = MLP(self.state_dim + self.action_dim * self.chunk_size + 1 , self.action_dim * self.chunk_size, self.hidden_dims)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        A_0 = torch.randn_like(action_chunk, device=action_chunk.device)
        target_velocity = action_chunk - A_0
        Batch_size = action_chunk.shape[0]


        tau = torch.rand((action_chunk.shape[0], 1, 1), device=action_chunk.device)
        A_tau = tau * action_chunk + (1 - tau) * A_0

        
        tau_flat = tau.reshape(Batch_size, 1)
        A_tau_flat = A_tau.reshape(Batch_size, -1)

        model_input = torch.cat((state, A_tau_flat, tau_flat), dim=1)
        pred_velocity = self.model(model_input).reshape(Batch_size, self.chunk_size,self.action_dim)
        mse = MSE()
        return mse.loss(first=pred_velocity, second=target_velocity)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:

        Batch_size = state.shape[0]

        A_tau = torch.randn(state.shape[0], self.chunk_size, self.action_dim)
        A_tau = A_tau.to(device=state.device)
        A_tau_flat = A_tau.reshape(Batch_size, -1)

        step_len = 1 / num_steps


        for step in range(num_steps):
            tau_flat = torch.full(size=(Batch_size, 1), fill_value=step / num_steps, device=state.device)
            model_input = torch.cat((state, A_tau_flat, tau_flat), dim=1)
            pred_velocity_chunk = self.model(model_input).reshape(state.shape[0], -1)
            A_tau_flat = A_tau_flat + step_len * pred_velocity_chunk

        return A_tau_flat.reshape(Batch_size, self.chunk_size,self.action_dim)

  



PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
