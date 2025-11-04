import torch
from torch.utils.data import Dataset
import h5py
from typing import Tuple
from common.img_process import preprocess_carracing_image


class PredictorDataset(Dataset):
    """
    PyTorch Dataset for data stored in HDF5 format.
    
    This dataset provides on-demand access to observations, actions, rewards,
    and done flags stored in the consolidated HDF5 file. Images are preprocessed
    to grayscale and properly normalized.
    
    Each sample consists of sequence data instead of single frames.
    """
    
    def __init__(
        self,
        h5_path: str = 'car_racing_data.h5',
        to_grayscale: bool = True
    ):
        """
        Initialize the CarRacing dataset.
        
        Args:
            h5_path: Path to the HDF5 file containing the dataset
            to_grayscale: Whether to convert RGB images to grayscale (default: True)
        """
        self.h5_path = h5_path
        self.to_grayscale = to_grayscale
        
        # Open the HDF5 file to retrieve dataset dimensions
        with h5py.File(self.h5_path, 'r') as h5f:
            self.total_episodes = h5f['images'].shape[0]
            self.max_steps = h5f['images'].shape[1]
        
        self.total_frames = self.total_episodes * self.max_steps
        self.h5_file = None  # Will be opened on first access
    
    def __len__(self) -> int:
        """Return the total number of frames in the dataset."""
        return self.total_frames
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve an episode sample from the dataset.
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        
        images = self.h5_file['images'][idx]
        action = self.h5_file['actions'][idx]
        reward = self.h5_file['rewards'][idx]
        done = self.h5_file['dones'][idx]
        
        images = torch.stack([preprocess_carracing_image(image) for image in images])
        next_images = images[1:]
        images = images[:-1]
        
        # Convert action, reward, and done to tensors
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        
        return images, action, reward, done, next_images
    
    def __del__(self):
        """Ensure the HDF5 file is closed when the dataset is deleted."""
        if self.h5_file is not None:
            self.h5_file.close()