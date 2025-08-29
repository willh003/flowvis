
import torch
from torch import nn, Tensor
from typing import List

from torch.utils.data import DataLoader, Dataset

import numpy as np

from pydrake.all import (
    PiecewisePolynomial,
    Trajectory,
)


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


def train_flow(model: ConditionalFlow, trajectories: List[Trajectory]) -> Tensor:
    dataset = TrajectoryDataset(trajectories)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for batch in dataloader:
        optimizer.zero_grad()
        loss = model.forward_train(batch)
        loss.backward()
        optimizer.step()


class Bottleneck1D(nn.Module):
    def __init__(self, cond_dim: int, q_dim: int, mid_dim: int):
        super().__init__()
        self.cond_dim = cond_dim
        self.q_dim = q_dim
        self.mid_dim = mid_dim

        self.velocity_field = nn.Sequential(
            nn.Linear(cond_dim+q_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, q_dim),
        )
    
    def forward(self, cond: Tensor, time: Tensor) -> Tensor:
        return self.velocity_field(cond, time)

class ConditionalFlow:

    def __init__(self, cond_dim: int, q_dim: int, prior: List[float]):
        self.cond_dim = cond_dim
        self.q_dim = q_dim
        self.prior = prior
        self.loss_fn = nn.MSELoss()
        self.velocity_field = self.make_nets(cond_dim, q_dim)
    
    def make_nets(self, cond_dim: int, q_dim: int, mid_dim: int=32) -> nn.Module:
        return Bottleneck1D(cond_dim, q_dim, mid_dim)

    def forward_train(self, posterior: Tensor, global_cond: Tensor, time: Tensor) -> Tensor:
        batch_size = posterior.shape[0]

        ts = torch.rand(batch_size, 1)
        prior = self.prior(batch_size)

        interp = ts * prior + (1 - ts) * posterior

        v = self.get_velocity(interp, global_cond, ts)

        target = posterior - prior

        loss = self.loss_fn(v, target)

        return loss


    def forward_inference(self, global_cond: Tensor) -> Tensor:
        batch_size = global_cond.shape[0]

        xt = self.prior(batch_size)
        for t in range(self.num_inference_steps):
            dt = 1 / self.num_inference_steps
            xt1 = xt + self.get_velocity(xt, global_cond) * dt

            xt = xt1

        return xt


    def get_velocity(self, cond: Tensor, time: Tensor) -> Tensor:
        return self.velocity_field(cond, time)


    def integrate_velocity(self, cond: Tensor, time: Tensor) -> Tensor:
        pass
