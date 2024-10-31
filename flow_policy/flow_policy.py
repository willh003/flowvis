import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from typing import List

from pydrake.all import PiecewisePolynomial, Trajectory


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

    def u_conditional(self, t: np.ndarray) -> np.ndarray:
        """
        Compute the conditional velocity field for each of the K trajectories.
        Args:
            t (np.ndarray, dtype=float, shape=(*BS,)): Time values in [0,1].
            
        Returns:
            (np.ndarray, dtype=float, shape=(*BS, K)): Velocities for each of the
                K conditional flows.
        """
        shape = t.shape
        t = t.ravel()  # (N,)
        us = []
        for traj in self.trajectories:
            traj_derivative = traj.MakeDerivative()
            u = traj_derivative.vector_values(t).reshape(shape)  # (*BS)
            us.append(u)

        us = np.array(us)  # (K, *BS)
        us = np.moveaxis(us, 0, -1)  # (*BS, K)
        return us

    def u_marginal(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Args:
            x (np.ndarray, dtype=float, shape=(*BS,)): State values.
            t (np.ndarray, dtype=float, shape=(*BS,)): Time values in [0,1].
            
        Returns:
            (np.ndarray, dtype=float, shape=(*BS,)): Marginal velocities.
        """
        likelihood = self.pdf_conditional(x, t)  # (*BS, K)
        posterior = self.π * likelihood  # (*BS, K)
        normalizing_constant = np.sum(posterior, axis=-1, keepdims=True)  # (*BS, 1)
        posterior = posterior / normalizing_constant  # (*BS, K)

        us = self.u_conditional(t)  # (*BS, K)
        return np.sum(us * posterior, axis=-1)  # (*BS,)

    def ode_integrate(self, x: float, num_steps: int = 1000) -> Trajectory:
        """
        Args:
            x (float): Initial state.
            num_steps (int): Number of steps to integrate.
            
        Returns:
            Trajectory: Trajectory starting from x.
        """
        breaks = np.linspace(0.0, 1.0, num_steps + 1)  # (N+1,)
        Δt = 1.0 / num_steps
        samples = [x]
        for t in breaks[:-1]:
            u = self.u_marginal(x, t)
            x = x + Δt * u
            samples.append(x)
        return PiecewisePolynomial.FirstOrderHold(breaks, [samples])
