
import torch
from torch import nn, Tensor
from typing import List, Union, Optional
from abc import ABC, abstractmethod
import math

from torch.utils.data import DataLoader, Dataset

import numpy as np

from .nets import Bottleneck1D, ConditionalUnet1D

from pydrake.all import (
    PiecewisePolynomial,
    Trajectory,
)
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal


def create_dummy_trajs(n_trajs: int = 100, std: float = 0.01, direction: str = 'right') -> List[Trajectory]:
    def demonstration_traj(std, direction) -> Trajectory:

        samples = [[0.00, 0.6, 0.25, 0.6, .2],
                   [0.00, 0.25, 0.5, 0.75, 1.00],
                   ]

        if direction == 'left':
            samples[0, :] = -samples[0, :]

        samples = np.array(samples)
        samples += np.random.normal(0, std, samples.shape)

        return PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            breaks=[0.00, 0.25, 0.50, 0.75, 1.0],
            samples=samples,
        )

    for i in range(n_trajs):
        yield demonstration_traj(std, direction)
    


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories: List[Trajectory], steps_per_traj: int = 100):
        self.trajectories = trajectories
        self.steps_per_traj = steps_per_traj
        self.all_states, self.all_actions = self.make_dataset(trajectories, steps_per_traj)


    def __len__(self):
        return len(self.all_states)

    def __getitem__(self, idx):
        return self.all_states[idx], self.all_actions[idx]

    def make_dataset(self, trajectories: List[Trajectory], steps_per_traj: int = 100) -> Dataset:
        times = np.linspace(0, 1, steps_per_traj)

        all_states = []
        all_actions = []

        for traj in trajectories:
            states = traj.vector_values(times)
            actions = states[:, 1:] - states[:, :-1]
            states = states[:, :-1]

            all_states.append(torch.from_numpy(states, dtype=torch.float32))
            all_actions.append(torch.from_numpy(actions, dtype=torch.float32))

        all_states, all_actions = torch.cat(all_states, dim=0), torch.cat(all_actions, dim=0)

        return all_states, all_actions




class GenerativePolicy(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward_train(self, posterior_samples: Tensor, cond: Tensor) -> Tensor:
        pass

    @abstractmethod
    def forward_inference(self, cond: Tensor) -> Tensor:
        pass
    
    @property
    @abstractmethod
    def nets(self) -> nn.Module:
        pass



class ConditionalFlow:

    def __init__(self, cond_dim: int, q_dim: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cond_dim = cond_dim
        self.q_dim = q_dim
        self.prior = self.make_uniform_prior()
        self.loss_fn = nn.MSELoss()
        self.velocity_field = self.make_nets(cond_dim, q_dim)

    @property
    def nets(self) -> nn.Module:
        return self.velocity_field


    def make_uniform_prior(self) -> Tensor:
        low = torch.zeros(self.q_dim)
        high = torch.ones(self.q_dim)
        return Uniform(low, high)

    def make_nets(self, cond_dim: int, q_dim: int) -> nn.Module:
        return ConditionalUnet1D(q_dim, cond_dim, diffusion_step_embed_dim=32, down_dims=[32,64]).to(self.device)

    def forward_train(self, states: Tensor, actions: Tensor) -> Tensor:

        global_cond = states.to(self.device)
        posterior_samples = actions.to(self.device)

        batch_size, chunk_size, posterior_dim = posterior_samples.shape

        self.chunk_size = chunk_size

        ts = torch.rand(batch_size, 1, 1).to(self.device)

        prior_samples = self.prior.sample((batch_size,chunk_size)).to(self.device)

        interp = ts * prior_samples + (1 - ts) * posterior_samples

        v = self.get_velocity(interp, ts[:,0,0], global_cond)

        target = posterior_samples - prior_samples
        loss = self.loss_fn(v, target)

        return loss


    def forward_inference(self, global_cond: Tensor, num_inference_steps: int = 100) -> Tensor:
        batch_size = global_cond.shape[0]
        
        # Start from prior samples
        x = self.prior.sample((batch_size, self.chunk_size)).to(self.device)
        global_cond = global_cond.to(self.device)
        
        # Integration step size
        dt = 1.0 / num_inference_steps
        
        # Integrate velocity field from t=0 to t=1 using Euler method
        with torch.no_grad():
            for step in range(num_inference_steps):
                t = torch.full((batch_size,), step * dt, device=self.device)
                v = self.get_velocity(x, t, global_cond)
                x = x + dt * v
        
        return x


    def get_velocity(self, x: Tensor, time: Tensor, global_cond: Tensor) -> Tensor:
        return self.velocity_field(x, time, global_cond)

