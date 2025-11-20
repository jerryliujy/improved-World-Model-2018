import torch
from torch.utils.data import DataLoader, Subset
from typing import Literal, Tuple
from common.img_process import preprocess_breakout_image

    
def get_dataloaders(
    dataset,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from a single CarRacing dataset.
    
    ✨ Key Advantage: Only ONE dataset object is created, and Subset is used to create
    different DataLoaders for each split. This avoids creating multiple datasets that
    each open the same HDF5 file separately.
    
    Args:
        h5_path: Path to the HDF5 file containing the dataset
        batch_size: Batch size for all DataLoaders
        train_ratio: Fraction of episodes for training (default: 0.7)
        val_ratio: Fraction of episodes for validation (default: 0.15)
        test_ratio: Fraction of episodes for testing (default: 0.15)
        to_grayscale: Whether to convert RGB images to grayscale (default: True)
        num_workers: Number of workers for DataLoader
        shuffle_train: Whether to shuffle training set (default: True)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"
    
    total_frames = len(dataset)
    
    train_size = int(total_frames * train_ratio)
    val_size = int(total_frames * val_ratio)
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_frames))
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader