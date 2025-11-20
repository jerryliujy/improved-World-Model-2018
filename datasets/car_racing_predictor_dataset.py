import torch
from torch.utils.data import Dataset
import h5py
from typing import Tuple
from common.img_process import preprocess_carracing_image


class CarRacingPredictorDataset(Dataset):
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
        
        self.h5_file = None  # Will be opened on first access
        self.z_encoded = []

    def preencode_images(self, vision_model, device='cuda'):
        """Pre-encode all images to z and load all actions"""
        with h5py.File(self.h5_path, 'r') as h5f:
            with torch.no_grad():
                for episode_idx in range(self.total_episodes):
                    images = h5f['images'][episode_idx]  # [steps, H, W, C]
                    
                    # Preprocess and encode
                    img_tensors = torch.stack([
                        preprocess_carracing_image(img, to_grayscale=self.to_grayscale)
                        for img in images
                    ])
                    img_tensors = img_tensors.to(device)
                    vision_model = vision_model.to(device)
                    
                    # Encode to z
                    z = vision_model.encode(img_tensors)  # [steps, latent_dim]
                    self.z_encoded.append(z.cpu())
    

    def __len__(self) -> int:
        """Return the total number of frames in the dataset."""
        return self.total_episodes
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve an episode sample from the dataset.
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        
        z_encoded = self.z_encoded[idx]
        action = self.h5_file['actions'][idx]
        reward = self.h5_file['rewards'][idx]
        done = self.h5_file['dones'][idx]
        
        next_z = z_encoded[1:]
        z = z_encoded[:-1]
        
        # Convert action, reward, and done to tensors
        action = torch.tensor(action[:-1], dtype=torch.float32)
        reward = torch.tensor(reward[:-1], dtype=torch.float32)
        done = torch.tensor(done[:-1], dtype=torch.float32)
        
        return z, action, reward, done, next_z
    
    def __del__(self):
        """Ensure the HDF5 file is closed when the dataset is deleted."""
        if self.h5_file is not None:
            self.h5_file.close()