import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mdnrnn import MDN, sample_mdn, gaussian_nll_loss
        
class MDNTransformer(nn.Module):
    def __init__(self, 
                 latent_dim=256,
                 num_heads=8,
                 num_layers=6,
                 num_gaussians=5):
        super(MDNTransformer, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim*4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.mdn = MDN(
            latent_dim=latent_dim,
            hidden_dim=latent_dim,
            num_gaussians=num_gaussians
        )
        
        def forward(self, z, a, tau=1.0):
            x = torch.cat([z, a], dim=-1)
            outputs = self.transformer_encoder(x)
            pi, mu, sigma = self.mdn(outputs, tau)
            return pi, mu, sigma
        
        def loss(self, pi, mu, sigma, z_next):
            return gaussian_nll_loss(pi, mu, sigma, z_next)