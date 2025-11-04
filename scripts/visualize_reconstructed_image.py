import matplotlib.pyplot as plt
import torch
import sys
import os
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.img_process import preprocess_carracing_image

from common.model_loader import load_checkpoint
from models.vae import VAE
from models.vq_vae import VQVAE
from datasets.datasets import CarRacingDataset

def plot_images(original, reconstructed, n=4, save_path='outputs/imgs/reconstructed_images.png'):
    fig, axes = plt.subplots(2, n, figsize=(12, 7))
    
    for i in range(n):
        axes[0, i].imshow(original[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].axis('off')
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold', loc='left')

    for i in range(n):
        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).cpu().detach().numpy())
        axes[1, i].axis('off')
    axes[1, 0].set_title("Reconstructed Images", fontsize=12, fontweight='bold', loc='left')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Reconstructed images saved to {save_path}")
    plt.show()
    
    
if __name__ == "__main__":
    h5_path = 'outputs/data/car_racing_data.h5'
    n = 4  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VAE(image_channels=3, latent_dim=32)
    load_checkpoint(model, "checkpoints/stage_1_epoch_0003.pth")
    model = model.to(device)
    model.eval()

    dataset = CarRacingDataset(h5_path=h5_path, to_grayscale=False)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=0
    )
    
    images, _, _, _, _ = next(iter(dataloader))
    images = images.to(device)

    with torch.no_grad():
        reconstructed, _, _ = model(images.squeeze())
    
    indices = torch.randperm(images.size(0))[:n]  # Select random images
    plot_images(images[indices], reconstructed[indices], n)