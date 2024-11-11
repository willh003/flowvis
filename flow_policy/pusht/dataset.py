import numpy as np
import torch
import zarr

from .dp_state_notebook import (
    create_sample_indices, get_data_stats, normalize_data, PushTStateDataset,
)

class PushTStateObsDatasetWithObsAsActions (PushTStateDataset):
    """
    A dataset of where gripper observations act as actions.
    """
    def __init__(
            self,
            dataset_path: str,
            pred_horizon: int,
            obs_horizon: int,
            action_horizon: int,
        ):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')

        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]

        train_data = self.MakeTrainData(dataset_root, episode_ends)

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    @staticmethod
    def MakeTrainData(dataset_root, episode_ends):
        # All demonstration episodes are concatinated in the first dimension N
        states = dataset_root['data']['state'][:]
        # Initialize a list to store the sliced and shifted episodes
        shifted_episodes = []
        # Iterate through the episode ends to slice and shift the tensor
        start_idx = 0
        for end_idx in episode_ends:
            episode = states[start_idx:end_idx]
            shifted_episode = np.concatenate((episode[1:], episode[-1].reshape(1, -1)), axis=0)
            shifted_episodes.append(shifted_episode)
            start_idx = end_idx
        

        # Combine the shifted episodes back into a single tensor if needed
        shifted_states = np.concatenate(shifted_episodes, axis=0)
        train_data = {
            # (N, action_dim)
            'action': shifted_states[:, :2],
            # (N, obs_dim)
            'obs': dataset_root['data']['state'][:]
        }

        return train_data