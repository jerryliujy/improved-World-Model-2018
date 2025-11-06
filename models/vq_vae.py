import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer layer for VQ-VAE.
    
    Args:
        num_embeddings: Number of vectors in the codebook
        embedding_dim: Dimensionality of each embedding vector
        commitment_cost: Scalar weight for commitment loss
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings uniformly in [-1/N, 1/N]
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, x):
        """
        Quantize input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, ..., embedding_dim]
            
        Returns:
            quantized: Quantized tensor (same shape as input)
            loss: VQ loss (scalar)
            perplexity: Perplexity of the codebook usage
            encodings: One-hot encodings for each input [batch_size, ..., num_embeddings]
        """
        # Flatten input keeping last dimension
        flat_x = x.view(-1, self.embedding_dim)
        
        # Calculate distances to all embeddings
        # ||x - e||^2 = ||x||^2 + ||e||^2 - 2<x, e>
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_x, self.embedding.weight.t())
        )
        
        # Get nearest embedding indices
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Create one-hot encodings
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices.view(-1, 1), 1)
        
        # Quantize and reshape
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(x.shape)
        
        # Calculate VQ loss
        # Commitment loss: encourages the encoder output to stay close to the chosen codebook vector
        commitment_loss = F.mse_loss(quantized.detach(), x, reduction='mean')
        
        # Codebook loss: encourages the codebook vectors to move towards the encoder outputs
        codebook_loss = F.mse_loss(quantized, x.detach(), reduction='mean')
        
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: copy gradients from quantized to x
        quantized = x + (quantized - x).detach()
        
        return quantized, vq_loss


class VQVAE(nn.Module):
    def __init__(self, 
                 image_channels=3, 
                 latent_dim=32,
                 num_embeddings=256,
                 commitment_cost=0.25):
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
            
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, self.latent_dim)
        )
        
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 6 * 6),
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
        
        self.vq_loss = None
        

    def encode(self, x):
        # Get encoder output
        z_pre_vq = self.encoder(x)  # [batch_size, latent_dim]
        
        z_quantized, _ = self.vector_quantizer(z_pre_vq)
        
        return z_quantized  # [batch_size, latent_dim]
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z_encoded = self.encoder(x)  # [batch_size, latent_dim]
        z_quantized, vq_loss = self.vector_quantizer(z_encoded)
        x_recon = self.decode(z_quantized)
        return x_recon, vq_loss

    def loss(self, recon, vq_loss, x, beta=0.25):
        recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')

        vq_loss = vq_loss
         
        total_loss = recon_loss + beta * vq_loss
        
        return total_loss