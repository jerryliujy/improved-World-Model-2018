import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mdnrnn import MDN, gaussian_nll_loss
        
class MDNTransformer(nn.Module):
    def __init__(self, 
                 latent_dim=32,
                 action_dim=3,
                 hidden_dim=256,
                 num_heads=8,
                 num_layers=6,
                 num_gaussians=5):
        super(MDNTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.proj_input = nn.Linear(latent_dim + action_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.mdn = MDN(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,  # transformer output dim
            num_gaussians=num_gaussians  
        )

    def _create_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def forward(self, z, a, tau=1.0, h=None):
        seq_len = z.size(1)
        causal_mask = self._create_causal_mask(seq_len)
        causal_mask = causal_mask.to(z.device)
        x = torch.cat([z, a], dim=-1)  # [batch_size, seq_len, latent_dim + action_dim]
        x = self.proj_input(x)  # [batch_size, seq_len, hidden_dim]
        outputs = self.transformer_encoder(
            x, 
            mask=causal_mask,
            is_causal=True
        )
        pi, mu, sigma = self.mdn(outputs, tau)
        h = outputs[:, -1, :].unsqueeze(0)
        return pi, mu, sigma, h  # h: transformer hidden states at the last time step 
    
    def loss(self, pi, mu, sigma, z_next):
        return gaussian_nll_loss(pi, mu, sigma, z_next)