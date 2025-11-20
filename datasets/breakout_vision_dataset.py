import io
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py
from typing import List, Tuple
from bisect import bisect_right
from PIL import Image
from common.img_process import preprocess_breakout_image


class _BreakoutEpisodeIndex:
    """Utility to map flat frame indices to variable-length episodes."""

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.episode_keys: List[str] = []
        self.episode_lengths: List[int] = []
        self.cumulative_lengths: List[int] = []
        self._load_metadata()

    def _load_metadata(self):
        with h5py.File(self.h5_path, 'r') as h5f:
            self.episode_keys = sorted(key for key in h5f.keys() if key.startswith('episode_'))
            total = 0
            for key in self.episode_keys:
                grp = h5f[key]
                length = grp['actions'].shape[0]
                self.episode_lengths.append(length)
                total += length
                self.cumulative_lengths.append(total)

        self.total_frames = total

    def locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0 or idx >= self.total_frames:
            raise IndexError(idx)
        episode_idx = bisect_right(self.cumulative_lengths, idx)
        prev_total = 0 if episode_idx == 0 else self.cumulative_lengths[episode_idx - 1]
        step = idx - prev_total
        return episode_idx, step
    

def _decode_observation(obs_entry: np.ndarray) -> np.ndarray:
    """Observations are stored as encoded bytes; decode if needed."""
    if obs_entry.ndim == 3:  # already HWC
        return obs_entry
    buffer = io.BytesIO(obs_entry.tobytes())
    image = Image.open(buffer)
    return np.array(image, dtype=np.uint8)
    
    
class BreakoutVisionDataset(Dataset):
    """Frame-wise dataset"""

    def __init__(
        self,
        h5_path: str = 'outputs/data/breakout_data.hdf5',
        to_grayscale: bool = True,
        one_hot_actions: bool = True,
        num_actions: int = 4,
    ):
        self.h5_path = h5_path
        self.to_grayscale = to_grayscale
        self.one_hot_actions = one_hot_actions
        self.num_actions = num_actions
        self.index = _BreakoutEpisodeIndex(h5_path)
        self.h5_file = None

    def __len__(self) -> int:
        return self.index.total_frames

    def __getitem__(self, idx: int):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        episode_idx, step = self.index.locate(idx)
        episode_key = self.index.episode_keys[episode_idx]
        grp = self.h5_file[episode_key]

        obs = grp['observations'][step]
        image = preprocess_breakout_image(_decode_observation(obs))

        obs_count = grp['observations'].shape[0]
        if step + 1 < obs_count:
            next_obs = grp['observations'][step + 1]
            next_image = preprocess_breakout_image(_decode_observation(next_obs))
        else:
            next_image = torch.zeros_like(image)

        action_value = int(grp['actions'][step])
        action = self._format_action(action_value)
        reward = torch.tensor(grp['rewards'][step], dtype=torch.float32)
        done_flag = bool(grp['terminations'][step]) 
        done = torch.tensor(float(done_flag), dtype=torch.float32)

        return image, action, reward, done, next_image

    def _format_action(self, action_value: int) -> torch.Tensor:
        if not self.one_hot_actions:
            return torch.tensor(action_value, dtype=torch.float32).unsqueeze(0)
        one_hot = torch.zeros(self.num_actions, dtype=torch.float32)
        if 0 <= action_value < self.num_actions:
            one_hot[action_value] = 1.0
        return one_hot

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()