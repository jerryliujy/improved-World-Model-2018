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




def get_breakout_dataloader(
    dataset, 
    batch_size: int = 128,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get a DataLoader for Breakout environment from Minari dataset.
    
    Args:
        batch_size: Batch size for DataLoader
        split: Which split to load ('train', 'val', or 'test')
        train_ratio: Fraction of data for training (default: 0.7)
        val_ratio: Fraction of data for validation (default: 0.15)
        test_ratio: Fraction of data for testing (default: 0.15)
        
    Returns:
        DataLoader for the specified split
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"
    
    def collate_fn(batch):
        # Preprocess observations to grayscale and downsampled
        processed_obs = []
        for x in batch:
            # x.observations has shape (episode_length, 210, 160, 3)
            obs_list = []
            for obs in x.observations:
                processed_obs_frame = preprocess_breakout_image(obs)
                obs_list.append(processed_obs_frame)
            # Stack to (episode_length, 1, 96, 96)
            processed_obs.append(torch.stack(obs_list, dim=0))
        
        return {
            "id": torch.Tensor([x.id for x in batch]),
            "observations": torch.nn.utils.rnn.pad_sequence(
                processed_obs,
                batch_first=True
            ),
            "actions": torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.actions, dtype=torch.long) for x in batch],
                batch_first=True
            ),
            "rewards": torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.rewards, dtype=torch.float32) for x in batch],
                batch_first=True
            ),
            "terminations": torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.terminations, dtype=torch.bool) for x in batch],
                batch_first=True
            ),
            "truncations": torch.nn.utils.rnn.pad_sequence(
                [torch.as_tensor(x.truncations, dtype=torch.bool) for x in batch],
                batch_first=True
            )
        }
    
    # Calculate split indices at the EPISODE level to avoid data leakage
    total_episodes = dataset.__getitem__(0)[0].shape[0]
    max_steps = dataset.__getitem__(0)[0].shape[1]

    train_end_ep = int(total_episodes * train_ratio)
    val_end_ep = train_end_ep + int(total_episodes * val_ratio)
    
    # Convert episode indices to frame indices
    train_indices = [i for i in range(len(dataset)) if (i // max_steps) < train_end_ep]
    val_indices = [i for i in range(len(dataset)) 
                   if train_end_ep <= (i // max_steps) < val_end_ep]
    test_indices = [i for i in range(len(dataset)) if (i // max_steps) >= val_end_ep]
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader