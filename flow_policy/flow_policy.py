import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from typing import List

from pydrake.all import Trajectory


class FlowPolicy:
    def __init__(
        self,
        trajectories: List[Trajectory],
        prior: List[float],
        sigma: float,
    ):
        """
        Args:
            trajectories (List[Trajectory]): List of trajectories.
            prior (np.ndarray, dtype=float, shape=(K,)): Prior
                probabilities for each trajectory.
            sigma (float): Standard deviation of the Gaussian distribution.
        """
        self.trajectories = trajectories
        self.π = np.array(prior)  # (K,)
        self.σ = sigma

    def μs(self, t: np.ndarray) -> np.ndarray:
        """
        Compute the mean of the conditional flows at time t
        
        Args:
            t (np.ndarray, dtype=float, shape=(*BS,)): Time values in [0,1].
            
        Returns:
            List[np.ndarray, dtype=float, shape=(*BS)]: Means of the conditional
                flows at time t.
        """
        shape = t.shape
        t = t.ravel()  # (N,)
        μs = []
        for traj in self.trajectories:
            μ = traj.vector_values(t).reshape(shape)  # (*BS,)
            μs.append(μ)
        return μs


    def log_pdf_marginal(self, x: np.ndarray, t: np.ndarray) -> float:
        """Compute log probability of the marginal flow at state x and time t
        
        Args:
            x (np.ndarray, dtype=float, shape=(*BS,)): State values.
            t (np.ndarray, dtype=float, shape=(*BS,)): Time values in [0,1].
            
        Returns:
            Log probabilities, shape=(*BS,)
        """
        μs = self.μs(t)  # List[(*BS)]

        log_probs = []
        for μ in μs:
            dist = norm(loc=μ, scale=self.σ)  # BS=(*BS) ES=()
            log_prob = dist.logpdf(x)  # (*BS,)
            log_probs.append(log_prob)
    
        log_probs = np.array(log_probs)  # (K, *BS)
        log_probs = np.moveaxis(log_probs, 0, -1)  # (*BS, K)
        log_probs = log_probs + np.log(self.π)  # (*BS, K)
        log_probs = logsumexp(log_probs, axis=-1)  # (*BS,)
        return log_probs

    def pdf_conditional(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Compute probability of the conditional flow at state x and time t, for
        each of the K trajectories.
        
        Args:
            x (np.ndarray, dtype=float, shape=(*BS,)): State values.
            t (np.ndarray, dtype=float, shape=(*BS,)): Time values in [0,1].
            
        Returns:
            (np.ndarray, dtype=float, shape=(*BS, K)): densities under each of
                the K conditional flows.
        """
        μs = self.μs(t)  # List[(*BS)]

        probs = []
        for μ in μs:
            dist = norm(loc=μ, scale=self.σ)  # BS=(*BS) ES=()
            prob = dist.pdf(x)  # (*BS,)
            probs.append(prob)
    
        probs = np.array(probs)  # (K, *BS)
        probs = np.moveaxis(probs, 0, -1)  # (*BS, K)
        return probs

    def pdf_marginal(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Compute probability of the marginal flow at state x and time t
        
        Args:
            x (np.ndarray, dtype=float, shape=(*BS,)): State values.
            t (np.ndarray, dtype=float, shape=(*BS,)): Time values in [0,1].
            
        Returns:
            Probabilities, shape=(*BS,)
        """
        probs = self.pdf_conditional(x, t)  # (*BS, K)
        probs = probs * self.π  # (*BS, K)
        probs = np.sum(probs, axis=-1)  # (*BS,)
        return probs

    def u(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Args:
            x: State values, shape=(*BS,)
            t: Time value in [0,1]
            
        Returns:
            Tensor: Action values, shape=(*BS, action_dim)
        """
        # Implement the action policy based on the state x and time t
        # This is a placeholder and should be replaced with the actual policy implementation
        return torch.zeros(x.shape[0], self.action_dim)
