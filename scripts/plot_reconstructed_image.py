import matplotlib.pyplot as plt
import torch
from common.img_process import preprocess_carracing_image
from common.model_loader import load_checkpoint
from models.vae import VAE
from models.vq_vae import VQVAE
from datasets.datasets import CarRacingDataset

def plot_images(original, reconstructed, n=4, save_path='outputs/img/reconstructed_images.png'):
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
    plt.show()
    plt.savefig(save_path)
    
    
if __name__ == "__main__":
    h5_path = 'car_racing_data.h5'
    n = 4  

    model = VAE(img_channels=3, latent_dim=32)
    checkpoint_path = 'checkpoints/stage_1_final.pth'
    load_checkpoint(model, checkpoint_path)
    model.eval()

    dataset = CarRacingDataset(h5_path=h5_path, to_grayscale=False)

    images = next(iter(dataset))[0].to('cuda').squeeze()  # Get a batch of images

    with torch.no_grad():
        reconstructed, _, _ = model(images)
    
    indices = torch.randperm(images.size(0))[:n]  # Select random images
    plot_images(images[indices], reconstructed[indices], n)