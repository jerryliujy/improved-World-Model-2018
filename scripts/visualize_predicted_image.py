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
from models.mdnrnn import MDNRNN, sample_mdn
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

    vision = VAE(image_channels=3, latent_dim=32)
    predictor = MDNRNN(
        latent_dim=32,  # latent_dim + action_dim
        action_dim=3,
        hidden_dim=256,
        num_gaussians=5,
        num_layers=1,
    )
    load_checkpoint(vision, "checkpoints/stage_1_epoch_0003.pth")
    load_checkpoint(predictor, "checkpoints/stage_2_epoch_0005.pth")
    
    vision = vision.to(device)
    predictor = predictor.to(device)
    vision.eval()
    predictor.eval()

    dataset = CarRacingDataset(h5_path=h5_path, to_grayscale=False)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=0
    )
    
    images, actions, _, _, next_images = next(iter(dataloader))
    images = images.to(device)
    actions = actions.to(device)
    z = vision.encode(images).unsqueeze(1)  # [batch, 1, latent_dim]

    with torch.no_grad():
        pi, mu, sigma, _ = predictor(z, actions.unsqueeze(1))
        z_next = sample_mdn(pi, mu, sigma)
        reconstructed = vision.decode(z_next.squeeze(1))

    
    indices = torch.randperm(images.size(0))[:n]  # Select random images
    plot_images(next_images[indices], reconstructed[indices], n)