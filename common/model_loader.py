import os
import torch

def save_checkpoint(model, dir, stage, epoch):
    """Save model checkpoint."""
    checkpoint_path = os.path.join(
        dir,
        f'stage_{stage}_epoch_{epoch + 1:04d}.pth'
    )
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')

def load_checkpoint(model, dir, stage, epoch):
    """Load model checkpoint."""
    checkpoint_path = os.path.join(
        dir,
        f'stage_{stage}_epoch_{epoch + 1:04d}.pth'
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    print(f'Checkpoint loaded from: {checkpoint_path}')