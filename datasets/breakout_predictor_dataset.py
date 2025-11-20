from typing import List
from datasets.breakout_vision_dataset import _BreakoutEpisodeIndex, _decode_observation
from common.img_process import preprocess_breakout_image
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class BreakoutPredictorDataset(Dataset):
    """Episode-wise dataset for predictor training using Breakout rollouts."""

    def __init__(
        self,
        h5_path: str = 'outputs/data/breakout_data.hdf5',
        to_grayscale: bool = True,
        num_actions: int = 4,
    ):
        self.h5_path = h5_path
        self.to_grayscale = to_grayscale
        self.num_actions = num_actions
        self.index = _BreakoutEpisodeIndex(h5_path)
        self.h5_file = None
        self.z: List[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.index.episode_keys)

    def preencode_images(self, vision_model, device='cuda'):
        """Pre-encode every observation into latent z using the trained vision model."""
        vision_model = vision_model.to(device)
        vision_model.eval()

        with torch.no_grad():
            with h5py.File(self.h5_path, 'r') as h5f:
                for key in self.index.episode_keys:
                    obs_ds = h5f[key]['observations']
                    frames = [
                        preprocess_breakout_image(_decode_observation(obs_ds[i]))
                        for i in range(len(obs_ds))
                    ]
                    imgs = torch.stack(frames, dim=0).to(device)
                    z = vision_model.encode(imgs)  # (steps, latent_dim)
                    self.z.append(z.cpu())

    def __getitem__(self, idx: int):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        episode_key = self.index.episode_keys[idx]
        grp = self.h5_file[episode_key]

        z_full = self.z[idx]
        z = z_full[:-1]
        next_z = z_full[1:]

        actions = torch.tensor(grp['actions'][:], dtype=torch.long)
        actions = F.one_hot(actions.clamp(min=0), num_classes=self.num_actions).float()
        rewards = torch.tensor(grp['rewards'][:], dtype=torch.float32)
        terminations = torch.tensor(grp['terminations'][:], dtype=torch.float32)
        truncations = torch.tensor(grp['truncations'][:], dtype=torch.float32)
        dones = torch.clamp(terminations + truncations, max=1.0)

        return z, actions, rewards, dones, next_z

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()