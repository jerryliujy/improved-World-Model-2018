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
        
        self.proj_z = nn.Linear(latent_dim, hidden_dim)
        self.proj_a = nn.Linear(action_dim, hidden_dim)
        
        # 0 -> z token, 1 -> a token
        self.type_embedding = nn.Embedding(2, hidden_dim)
        
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
    
    @staticmethod
    def _sinusoidal_pos_encoding(length, dim, device):
        # standard sinusoidal positional encoding
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)  # [L,1]
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [L, dim]
    
    def forward(self, z, a, tau=1.0, h=None):
        batch_size, seq_len = z.size(0), z.size(1)
        device = z.device
        
        z_proj = self.proj_z(z)
        a_proj = self.proj_a(a)
        
        # interleave z and a tokens: [z0, a0, z1, a1, ..., z_{seq_len-1}, a_{seq_len-1}]
        token_len = seq_len * 2
        tokens = torch.zeros(batch_size, token_len, self.hidden_dim, device=device)
        tokens[:, 0::2, :] = z_proj
        tokens[:, 1::2, :] = a_proj
        
        # positional embedding and type embedding
        pos_emb = self._sinusoidal_pos_encoding(token_len, self.hidden_dim, device)
        pos_emb = pos_emb.unsqueeze(0)
        type_ids = (torch.arange(token_len, device=device) % 2).long()
        type_emb = self.type_embedding(type_ids).unsqueeze(0)
        enc_input = tokens + pos_emb + type_emb
        
        # attn encoder
        causal_mask = self._create_causal_mask(token_len)
        causal_mask = causal_mask.to(device)
        outputs = self.transformer_encoder(
            enc_input, 
            mask=causal_mask,
            is_causal=True
        )
        
        # mdn head
        action_outputs = outputs[:, 1::2, :]
        pi, mu, sigma = self.mdn(action_outputs, tau)
        h = outputs[:, -1, :]
        return pi, mu, sigma, h  # h: transformer hidden states at the last time step 
    
    def loss(self, pi, mu, sigma, h, z_next):
        return gaussian_nll_loss(pi, mu, sigma, z_next)