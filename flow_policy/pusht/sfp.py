from typing import Dict, Optional
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from pydrake.all import PiecewisePolynomial
from torchdyn.core import NeuralODE

from flow_policy.pusht.dp_state_notebook.base_policy import Policy


class StreamingFlowPolicyPositionOnly (Policy):
    def __init__(self,
                 velocity_net: nn.Module,
                 action_dim: int,
                 pred_horizon: int,
                 sigma: float,
                 device: torch.device,
        ):
        """
        Args:
            velocity_net (nn.Module): velocity network
            action_dim (int): action dimension
            pred_horizon (int): prediction horizon
            sigma (float): standard deviation of the Gaussian noise
            device (torch.device): device
        """
        self.velocity_net = velocity_net
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.sigma = sigma
        self.device = device

    def TransformTrainingDatum(self, datum: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Args:
            datum (Dict[str, np.ndarray]):
                'obs' (np.ndarray, shape=(OBS_HORIZON, OBS_DIM), dtype=np.float32)
                'action' (np.ndarray, shape=(PRED_HORIZON, ACTION_DIM), dtype=np.float32)

        Returns:
            Dict[str, np.ndarray]:
                'obs' (np.ndarray, shape=(OBS_HORIZON, OBS_DIM), dtype=np.float32)
                'x' (np.ndarray, shape=(ACTION_DIM,), dtype=np.float32): position
                'u' (np.ndarray, shape=(ACTION_DIM,), dtype=np.float32): velocity
                't' (np.ndarray, shape=(,), dtype=np.float32): time
        """
        obs, action = datum['obs'], datum['action']
        OBS_HORIZON, OBS_DIM = obs.shape
        PRED_HORIZON, ACTION_DIM = action.shape
        assert PRED_HORIZON == self.pred_horizon
        assert OBS_HORIZON == 2  # logic currently only works for history of length 2

        # Create a trajectory from the action sequence.
        traj_times = np.linspace(0, 1, PRED_HORIZON)  # (PRED_HORIZON,)
        traj_positions = action  # (PRED_HORIZON, ACTION_DIM)
        traj: PiecewisePolynomial = PiecewisePolynomial.FirstOrderHold(
            traj_times, traj_positions.T,
        )

        # Ensure that the first action matches the last observation
        # This may not happen when the sequence starts from the beginning of an
        # episode and both obs and action are duplicated. Then, hack this by
        # setting the first action to the last observation. However, the right
        # thing to do is:
        # TODO (Sid): Set the first action correctly when creating the dataset.
        if not np.all(action[0] == obs[-1, :2]):
            action = action.copy()
            action[0] = obs[-1, :2]

        time = np.float32(np.random.rand())  # (,)
        x = traj.value(time).T  # (1, ACTION_DIM)
        u = traj.EvalDerivative(time).T  # (1, ACTION_DIM)

        # Add noise to position
        x = x + self.sigma * np.random.randn(*x.shape)  # (1, ACTION_DIM)
        x = x.astype(np.float32)  # (1, ACTION_DIM)

        return {
            'obs': obs,  # (OBS_HORIZON, OBS_DIM)
            'x': x.astype(np.float32),  # (1, ACTION_DIM)
            'u': u.astype(np.float32),  # (1, ACTION_DIM)
            't': time,  # (,)
        }

    @torch.enable_grad()
    def Loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            batch (Dict[str, Tensor]):
                'obs' (Tensor, shape=(B, OBS_HORIZON, OBS_DIM))
                'x' (Tensor, shape=(B, 1, ACTION_DIM))
                'u' (Tensor, shape=(B, 1, ACTION_DIM))
                't' (Tensor, shape=(B,))

        Returns:
            Tensor (shape=(,), dtype=torch.float32): loss
        """
        # device transfer
        obs = batch['obs'].to(self.device)  # (B, OBS_HORIZON, OBS_DIM)
        x = batch['x'].to(self.device)  # (B, 1, ACTION_DIM)
        u = batch['u'].to(self.device)  # (B, 1, ACTION_DIM)
        t = batch['t'].to(self.device)  # (B,)
        B = obs.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs.flatten(start_dim=1)  # (B, OBS_HORIZON * OBS_DIM)

        # predict the velocity
        u_pred = self.velocity_net(
            sample=x, timestep=t, global_cond=obs_cond
        )  # (B, ACTION_DIM)

        # L2 loss
        loss = nn.functional.mse_loss(u_pred, u)  # (,)
        return loss

    @torch.inference_mode()
    def __call__(self, obs_cond: Tensor, pred_horizon: int, num_actions: Optional[int] = None) -> Tensor:
        ode = NeuralODE(
            vector_field=VectorFieldWrapper(self.velocity_net, obs_cond),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        x0 = obs_cond[-1][:2]  # (2,)

        # Integration time steps
        integration_steps_per_action = 6
        num_actions = num_actions or pred_horizon
        tmax = pred_horizon / num_actions
        total_integration_steps = 1 + num_actions * integration_steps_per_action
        t_span = torch.linspace(0, tmax, total_integration_steps)
        select_action_indices = np.arange(
            integration_steps_per_action,
            total_integration_steps,
            integration_steps_per_action,
        )

        traj = ode.trajectory(x=x0, t_span=t_span)  # (total_integration_steps, 2)

        naction = traj[select_action_indices]  # (NUM_ACTIONS, 2)
        return naction


class VectorFieldWrapper (nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model: nn.Module, obs_cond: Tensor):
        super().__init__()
        self.model = model
        self.obs_cond = obs_cond

    def forward(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        return self.model(
            sample=x,
            timestep=t.repeat(x.shape[0]),
            global_cond=self.obs_cond.flatten(start_dim=1)
        )
