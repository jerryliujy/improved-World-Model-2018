import torch
import torch.nn as nn
import torch.nn.functional as F

class VQVAE(nn.Module):
    def __init__(self, 
                 image_channels=3, 
                 latent_dim=32,
                 num_embeddings=256):
        super(VQVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 3 x 96 x 96
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        self.encoder_output = nn.Linear(256 * 6 * 6, self.latent_dim)
        
        # latent codebook
        self.codebook = nn.Embedding(num_embeddings=256, embedding_dim=latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 256 * 6 * 6)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 6, 6)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        

    def encode(self, x):
        h = self.encoder_output(self.encoder(x))
        return self.codebook(h)
    
    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def loss(self, recon, mu, logvar, x, beta=2.0):
        batch_size = x.size(0)

        recon_loss = nn.functional.mse_loss(recon, x, reduction='sum') / batch_size
        
        kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / batch_size
         
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss