import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def clone(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([module for _ in range(N)])

def subsequent_mask(size):
    """Create a mask to hide future tokens."""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).byte()
    return subsequent_mask == 0

class Embedding(nn.Module):
    """Input Embedding layer for Transformer.
    """
    def __init__(self, vocal_size, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(vocal_size, embedding_dim)
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.embedding_dim)

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for Transformer.
    """
    def __init__(self, embedding_dim, max_len=512):
        super().__init__()
        self.embedding_dim = embedding_dim

        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1) 
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim)
        )  # (embedding_dim // 2,) 
        
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)  # fixed
        
    def forward(self, x):
        seq_len = x.size(1) 
        return self.pe[:seq_len]

class FeedForwardLayer(nn.Module):
    """Feed-forward layer for Transformer.
    """
    def __init__(self, embedding_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        x_prev = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.norm(x + x_prev)


class Attention(nn.Module):
    """Scaled dot-product attention.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        
    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embedding_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        scores = torch.softmax(scores, dim=-1)
        output = torch.matmul(scores, value)
        return output
        
        
class MultiHeadAttention(nn.Module):
    """Multi-head attention module.
    This module splits the input into multiple heads, applies scaled dot-product attention to each head,
    and concatenates the results.
    """
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert (
            self.head_dim * num_heads == embedding_dim
        ), "Embedding dimension must be divisible by number of heads"
        
        self.attention = Attention(embedding_dim)

        self.linear_q = nn.Linear(embedding_dim, embedding_dim)
        self.linear_k = nn.Linear(embedding_dim, embedding_dim)
        self.linear_v = nn.Linear(embedding_dim, embedding_dim)
        
        self.linear_out = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)
        
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (b, seq) -> (b, 1, 1, seq)
            
        output = self.attention(query, key, value, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        # output = [attn(query[i], key[i], value[i], mask) for i, attn in enumerate(self.attentions)]
        # output = torch.cat(output, dim=-1)
        output = self.linear_out(output)
        output = self.dropout(output)
        return output
    

class MultiHeadSelfAttnLayer(nn.Module):
    """Multi-head self-attention layer.
    """
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attns = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attns(x, x, x, mask)
        return self.norm(attn_output + x)

class MultiHeadCrossAttnLayer(nn.Module):
    """Multi-head cross-attention layer.
    """
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attns = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.embedding_dim = embedding_dim
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, query, key, mask=None):
        attn_output = self.cross_attns(
            query=query, 
            key=key, 
            value=key, 
            mask=mask)
        return self.norm(attn_output + query)

class EncoderLayer(nn.Module):
    """Encoder layer for Transformer.
    This module consists of a multi-head self-attention layer and a feed-forward layer.
    """
    def __init__(self, 
                 trans_dim, 
                 num_heads, 
                 ff_dim, 
                 dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttnLayer(
            embedding_dim=trans_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.feed_forward = FeedForwardLayer(
            embedding_dim=trans_dim,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
    def forward(self, x, mask=None):
        x = self.self_attn(x, mask)
        x = self.feed_forward(x)
        return x
    
class DecoderLayer(nn.Module):
    """Decoder layer for Transformer.
    This module consists of a multi-head self-attention layer, a multi-head cross-attention layer (with encoder output as k & v),
    and a feed-forward layer.
    """
    def __init__(self,
                 trans_dim,
                 num_heads,
                 ff_dim,
                 dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttnLayer(
            embedding_dim=trans_dim,
            num_heads=num_heads
        )
        self.cross_attn = MultiHeadCrossAttnLayer(
            embedding_dim=trans_dim,
            num_heads=num_heads
        )
        self.feed_forward = FeedForwardLayer(
            embedding_dim=trans_dim,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = self.self_attn(x, tgt_mask)
        x = self.cross_attn(x, memory, src_mask)
        x = self.feed_forward(x)
        return x


class Encoder(nn.Module):
    """Encoder module for Transformer.
    Stacks of encoder layers.
    """
    def __init__(self, 
                 trans_dim, 
                 num_heads, 
                 num_layers, 
                 vocab_size,
                 ff_dim,
                 dropout=0.1):
        super().__init__()
        self.input_embed = Embedding(vocal_size=vocab_size, embedding_dim=trans_dim)
        self.pos_embed = SinusoidalPositionalEmbedding(embedding_dim=trans_dim)
        self.layers = clone(EncoderLayer(
            trans_dim=trans_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout), num_layers)
        
    def forward(self, x, mask=None):
        x = self.input_embed(x) + self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    """Decoder module for Transformer.
    Stacks of decoder layers.
    """
    def __init__(self, 
                 trans_dim, 
                 num_heads, 
                 num_layers, 
                 vocab_size,
                 ff_dim,
                 dropout=0.1):
        super().__init__()
        self.input_embed = Embedding(vocal_size=vocab_size, embedding_dim=trans_dim)
        self.pos_embed = SinusoidalPositionalEmbedding(embedding_dim=trans_dim)
        self.layers = clone(DecoderLayer(
            trans_dim=trans_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout), num_layers)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.input_embed(x) + self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

class EncoderDecoder(nn.Module):
    """A standard Encoder-Decoder architecture.
    This module is a combination of an encoder and a decoder.
    It takes a source sequence and a target sequence as input,
    and produces an output sequence.
    """
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, src_mask, tgt_mask)
        return output


class Generator(nn.Module):
    def __init__(self, trans_dim, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(trans_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
        

class Transformer(nn.Module):
    def __init__(self,
                 trans_dim=512,
                 num_heads=8,
                 ff_dim=2048,
                 vocab_size=10000,
                 encoder_layers=6,
                 decoder_layers=8,
                 dropout=0.1,
                 max_len=512,
                 ):
        super(Transformer, self).__init__()
        encoder = Encoder(
            trans_dim=trans_dim,
            num_heads=num_heads,
            num_layers=encoder_layers,
            vocab_size=vocab_size,  
            ff_dim=ff_dim,
            dropout=dropout
        )
        decoder = Decoder(
            trans_dim=trans_dim,
            num_heads=num_heads,
            num_layers=decoder_layers,
            vocab_size=vocab_size,
            ff_dim=ff_dim,
            dropout=dropout   
        )
        self.encoder_decoder = EncoderDecoder(encoder, decoder)
        self.generator = Generator(trans_dim, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if tgt_mask is None:
            tgt_mask = subsequent_mask(tgt.size(1)).to(tgt.device)
            
        output = self.generator(self.encoder_decoder(src, tgt, src_mask, tgt_mask)) 
        return output