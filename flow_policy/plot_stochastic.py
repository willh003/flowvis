import numpy as np
import matplotlib.pyplot as plt
from typing import List

from flow_policy.stochastic_flow_policy import StochasticFlowPolicy

def plot_probability_density_q(
        fp: StochasticFlowPolicy,
        ts: np.ndarray,
        xs: np.ndarray,
        ax: plt.Axes,
        normalize: bool=True,
        alpha: float=1,
    ):
    p = np.zeros((len(ts), len(xs)))  # (T, X)
    for i in range(len(ts)):
        for j in range(len(xs)):
            p[i,j] = fp.pdf_marginal_q(xs[j], ts[i])

    if normalize:
        p = p / p.max(axis=1, keepdims=True)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)

    extent = [xs[0], xs[-1], ts[0], ts[-1]]
    return ax.imshow(p, origin='lower', extent=extent, aspect='auto', alpha=alpha)

def plot_probability_density_ε(
        fp: StochasticFlowPolicy,
        ts: np.ndarray,
        xs: np.ndarray,
        ax: plt.Axes,
        normalize: bool=True,
        alpha: float=1,
    ):
    p = np.zeros((len(ts), len(xs)))  # (T, X)
    for i in range(len(ts)):
        for j in range(len(xs)):
            p[i,j] = fp.pdf_marginal_ε(xs[j], ts[i])
    
    if normalize:
        p = p / p.max(axis=1, keepdims=True)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)

    extent = [xs[0], xs[-1], ts[0], ts[-1]]
    return ax.imshow(p, origin='lower', extent=extent, aspect='auto', alpha=alpha)


def plot_probability_density_and_vector_field(fp: StochasticFlowPolicy, ax: plt.Axes, num_points=200, num_quiver=20):
    """
    Example of how to call the function:

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    im = plot_probability_density_and_vector_field(fp, ax)
    plt.colorbar(im, ax=ax, label='Probability Density')
    plt.show()
    """
    ts = np.linspace(0, 1, num_points)  # (T,)
    xs = np.linspace(-1, 1, num_points)  # (X,)
    ts, xs = np.meshgrid(ts, xs, indexing='ij')  # (T, X)
    u = fp.u_marginal(xs, ts)  # (T, X)

    # Plot probability density
    heatmap = plot_probability_density(fp, ts, xs, ax)

    # Plot quiver with reduced size
    quiver_step_x = xs.shape[1] // num_quiver
    quiver_step_t = ts.shape[0] // num_quiver
    ax.quiver(
        xs[::quiver_step_t, ::quiver_step_x],
        ts[::quiver_step_t, ::quiver_step_x], 
        u[::quiver_step_t, ::quiver_step_x],
        np.ones_like(u)[::quiver_step_t, ::quiver_step_x], 
        color='white', scale=40, width=0.002, headwidth=3, headlength=4
    )

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Probability Density and Vector Field')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time ⟶')

    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    return heatmap

def plot_probability_density_and_streamlines(fp: StochasticFlowPolicy, ax: plt.Axes, num_points: int=400):
    """
    Example of how to call the function:

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    im = plot_probability_density_and_streamlines(fp, ax)
    plt.colorbar(im, ax=ax, label='Probability Density')
    plt.show()
    """
    ts = np.linspace(0, 1, num_points)  # (T,)
    xs = np.linspace(-1, 1, num_points)  # (X,)
    ts, xs = np.meshgrid(ts, xs, indexing='ij')  # (T, X)
    u = fp.u_marginal(xs, ts)  # (T, X)

    # Plot log probability
    heatmap = plot_probability_density(fp, ts, xs, ax)

    # Plot streamlines
    ax.streamplot(x=xs[0], y=ts[:, 0], u=u, v=np.ones_like(u), 
                  color='white', density=1, linewidth=0.5, arrowsize=0.5)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Probability Density and Flow')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time ⟶')

    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    return heatmap

def plot_probability_density_with_trajectories(
        fp: StochasticFlowPolicy,
        ax: plt.Axes,
        xs_start: List[float | None],
        linewidth: float=1,
        alpha: float=0.5,
        heatmap_alpha: float=1,
    ):
    ts = np.linspace(0, 1, 200)  # (T,)
    xs = np.linspace(-1, 1, 200)  # (X,)
    ts, xs = np.meshgrid(ts, xs, indexing='ij')  # (T, X)
    heatmap = plot_probability_density(fp, ts, xs, ax, alpha=heatmap_alpha)

    for x_start in xs_start:
        x_start = x_start if x_start is not None else np.random.randn() * fp.σ
        traj = fp.ode_integrate(x_start)
        ts = np.linspace(0, 1, 200)
        xs = traj.vector_values(ts)
        ax.plot(xs[0], ts, color='red', linewidth=linewidth, alpha=alpha)

    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Trajectories sampled from flow')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time ⟶')

    return heatmap
