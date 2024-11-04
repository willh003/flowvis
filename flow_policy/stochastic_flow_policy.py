import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from typing import List, Tuple

from pydrake.all import PiecewisePolynomial, Trajectory


class StochasticFlowPolicy:
    def __init__(
        self,
        trajectories: List[Trajectory],
        prior: List[float],
        σ0: float,
        σ1: float,
    ):
        """
        Flow policy is an extended configuration space (q(t), ε(t)) where q is
        the original trajectory and ε is a noise variable that starts from
        N(0, 1).

        Let q̃(t) be the demonstration trajectory.
        Define σt = (1-t)σ0 + tσ1.

        Conditional flow:
        • At time t=0, we sample:
            • q₀ ~ N(q̃(0), σ₀)
            • ε₀ ~ N(0, 1)

        • Flow trajectory at time t:
            • q(t) = q₀ + (q̃(t) - q̃(0)) + (σt - σ0) ε₀
            • ε(t) = (1-t)ε₀ + tq̃(t)
              • ε starts from a pure noise sample that drifts towards the
              trajectory. Therefore, ε(t) is uncorrelated with q at t=0, but
              eventually becomes very informative of the trajectory.

        Args:
            trajectories (List[Trajectory]): List of trajectories.
            prior (np.ndarray, dtype=float, shape=(K,)): Prior
                probabilities for each trajectory.
            σ0 (float): Standard deviation of the Gaussian tube at time t=0.
            σ1 (float): Standard deviation of the Gaussian tube at time t=1.
        """
        assert σ1 >= σ0

        self.trajectories = trajectories
        self.π = np.array(prior)  # (K,)
        self.σ0 = σ0
        self.σ1 = σ1


    def σ(self, t: float) -> float:
        """
        Returns:
            float: Standard deviation of the Gaussian tube at time t.
        """
        return (1-t) * self.σ0 + t * self.σ1


    def Ab(self, traj: Trajectory, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            A (np.ndarray, dtype=float, shape=(2, 2)): Transition matrix.
            b (np.ndarray, dtype=float, shape=(2,)): Bias vector.
        """
        σt = self.σ(t)  # (1,)
        σ0 = self.σ0  # (1,)
        q̃0 = traj.value(0)
        q̃t = traj.value(t)

        b = np.array([q̃t - q̃0, t * q̃t])
        A = np.array([
            [1, σt - σ0],
            [0,     1-t],
        ])
        return A, b

    def μΣ(self, traj: Trajectory, t: float) -> np.ndarray:
        """
        Compute the mean and covariance matrix of the conditional flows at time t.
        
        Args:
            traj (Trajectory): Demonstration trajectory.
            t (float): Time value in [0,1].
            
        Returns:
            np.ndarray, dtype=float, shape=(2,): Mean of extended configuration
                space of the conditional flow at time t.
        """
        q̃0 = traj.value(0)
        σ0 = self.σ0
        A, b = self.Ab(traj, t)
        μ0 = np.array([q̃0, 0])
        Σ0 = np.array([[σ0**2, 0], [0, 1]])
        μt = A @ μ0 + b
        Σt = A @ Σ0 @ A.T
        return μt, Σt

    def pdf_conditional(self, traj: Trajectory, x: np.ndarray, t: float) -> float:
        """
        Compute probability of the conditional flow at state x and time t, for
        each of the K trajectories.
        
        Args:
            traj (Trajectory): Demonstration trajectory.
            x (np.ndarray, dtype=float, shape=(2,)): State values.
            t (float): Time value in [0,1].
            
        Returns:
            float: Probability of the conditional flow at state x and time t.
        """
        μt, Σt = self.μΣ(traj, t)
        dist = multivariate_normal(mean=μt, cov=Σt)
        return dist.pdf(x)

    def pdf_marginal(self, x: np.ndarray, t: float) -> float:
        """
        Compute probability of the marginal flow at state x and time t
        
        Args:
            x (np.ndarray, dtype=float, shape=(2,)): State values.
            t (float): Time value in [0,1].
            
        Returns:
            float: Probability of the marginal flow at state x and time t.
        """
        prob = 0
        for π, traj in zip(self.π, self.trajectories):
            prob += π * self.pdf_conditional(traj, x, t)
        return prob

    def u_conditional(self, traj: Trajectory, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the conditional velocity field for a given trajectory.

        • Flow trajectory at time t:
            • q(t) = q₀ + (q̃(t) - q̃(0)) + (σt - σ0) ε₀
            • ε(t) = (1-t)ε₀ + tq̃(t)

        • Conditional velocity field:
            • First, given q(t) and ε(t), we want to compute q₀ and ε₀.
                • ε₀ = (ε(t) - tq̃(t)) / (1-t)
                • q₀ = q(t) - (q̃(t) - q̃(0)) - (σt - σ₀) ε₀
            • Then, we compute the velocity field for the conditional flow.
                • uq(q, ε, t) = ṽ(t) + (σt - σ0) ε₀
                • uε(q, ε, t) = tṽ(t) - ε₀

        Args:
            traj (Trajectory): Demonstration trajectory.
            x (np.ndarray, dtype=float, shape=(2,)): State values.
            t (float): Time value in [0,1].
            
        Returns:
            (np.ndarray, dtype=float, shape=(2)): Velocities for each of the
                K conditional flows.
        """
        qt, εt = x
        q̃0 = traj.value(0)
        q̃t = traj.value(t)
        σt = self.σ(t)
        σ0 = self.σ0

        ṽt = traj.EvalDerivative(t)

        # Invert the flow and transform (qt, εt) to (q0, ε0)
        ε0 = (εt - t * q̃t) / (1-t)
        q0 = qt - (q̃t - q̃0) - (σt - σ0) * ε0

        # Compute velocity of the trajectory starting from (q0, ε0) at t
        uq = ṽt + (σt - σ0) * ε0
        uv = t * ṽt - ε0

        return np.array([uq, uv])


    def u_marginal(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Args:
            x (np.ndarray, dtype=float, shape=(2,)): State values.
            t (float): Time value in [0,1].

        Returns:
            (np.ndarray, dtype=float, shape=(2,)): Marginal velocities.
        """
        likelihoods = np.hstack([self.pdf_conditional(traj, x, t) for traj in self.trajectories])  # (K,)
        velocities = np.vstack([self.u_conditional(traj, x, t) for traj in self.trajectories])  # (K, 2)

        posterior = self.π * likelihoods
        normalizing_constant = np.sum(posterior)  # (1,)
        posterior = posterior / normalizing_constant  # (K,)
        posterior = posterior.reshape(-1, 1)  # (K, 1)

        us = (posterior * velocities).sum(axis=0)  # (2,)
        return us

    def ode_integrate(self, x: np.ndarray, num_steps: int = 1000) -> Trajectory:
        """
        Args:
            x (np.ndarray, dtype=float, shape=(2,)): Initial state.
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
