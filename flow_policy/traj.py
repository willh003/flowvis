import dataclasses as dc
import torch
from torch import Tensor


@dc.dataclass
class Trajectory:
    """
    x: Lies in [-1, 1]
    t: Lies in [ 0, 1]. The start time is always 0 and  end time is always 1.
    """
    x: Tensor  # dtype=float, shape=(N,)
    t: Tensor  # dtype=float, shape=(N,)


def JoinTrajectories(traj1: Trajectory, traj2: Trajectory) -> Trajectory:
    """
    Concatenate two trajectories.
    """
    x = torch.cat([traj1.x, traj2.x])
    t = torch.cat([traj1.t, traj2.t])
    return Trajectory(x=x, t=t)
