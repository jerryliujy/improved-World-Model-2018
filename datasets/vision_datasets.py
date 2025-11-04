import torch
from torch.utils.data import Dataset
import h5py
from typing import Tuple
from common.img_process import preprocess_carracing_image


class VisionDataset(Dataset):
    """
    PyTorch Dataset for vision data stored in HDF5 format.
    
    This dataset provides on-demand access to observations, actions, rewards,
    and done flags stored in the consolidated HDF5 file. Images are preprocessed
    to grayscale and properly normalized.
    
    It returns single frames of data.
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
        Retrieve a single frame and associated data.
        
        Args:
            idx: Index of the frame to retrieve (0 to total_frames-1)
            
        Returns:
            tuple: (image, action, reward, done) tensors
                - image: torch.Tensor of shape (1, 96, 96) if grayscale
                - action: torch.Tensor of shape (3,) - [steering, acceleration, brake]
                - reward: torch.Tensor scalar
                - done: torch.Tensor scalar (boolean)
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        
        # Calculate episode and step from the flat index
        episode = idx // self.max_steps
        step = idx % self.max_steps
        
        # Access the datasets directly using episode and step indices
        image = self.h5_file['images'][episode, step]
        action = self.h5_file['actions'][episode, step]
        reward = self.h5_file['rewards'][episode, step]
        done = self.h5_file['dones'][episode, step]
        
        # Preprocess image
        image = preprocess_carracing_image(image, to_grayscale=self.to_grayscale)
        
        # Convert action, reward, and done to tensors
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        
        return image, action, reward, done
    
    def __del__(self):
        """Ensure the HDF5 file is closed when the dataset is deleted."""
        if self.h5_file is not None:
            self.h5_file.close()
