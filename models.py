import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

from layers import *


class VisionTransformer(nn.Module):
    """ Implementation of Dosovitskiy et als. Vision Transformer Architecture""" 
    def __init__(self, num_patches, input_dimension, latent_dimension, num_classes, num_encoder_blocks, num_heads):
        super().__init__()
        self.projection_layer = nn.Linear(input_dimension, latent_dimension)

        self.embedding_token = nn.Parameter(data = torch.empty(1, latent_dimension), requires_grad = True)
        init.kaiming_uniform_(self.embedding_token) 

        self.position_embeddings = nn.Embedding(num_patches + 1, latent_dimension)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(num_heads, latent_dimension, latent_dimension) for _ in range(num_encoder_blocks)])

        self.layer_norm = nn.LayerNorm(latent_dimension)

        self.classification_head = nn.Sequential(nn.Linear(latent_dimension, latent_dimension), nn.GELU(), nn.Linear(latent_dimension, num_classes))

    def forward(self, input):
        #Input shape: batch_size x num patches (i.e sequence length) x input_dimension 
        x = self.projection_layer(input)
        #x shape: batch_size x num patches (i.e sequence length) x latent_dimension
        embedding_token = self.embedding_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((embedding_token, x), dim = 1)
        #x shape: batch_size x num patches + 1 x latent_dimension
        pos_indices = torch.arange(x.shape[1], device=x.device)
        pos_embeddings = self.position_embeddings(pos_indices)
        #pos_embeddings shape: num patches + 1 x latent_dimension
        x = x + pos_embeddings 
        
        for encoder in self.encoder_blocks:
            x = encoder(x)
        #x shape: batch_size x num patches + 1 x latent_dimension  
        
        x_cls = torch.squeeze(x[:, 0, :], 1)
        #x_cls shape: batch_size x latent_dimension  
        x_cls = self.layer_norm(x_cls)
        output = self.classification_head(x_cls)
        #output shape: batch_size x num_classes
        return output 

 
class Other_VisionTransformer(nn.Module):
    """ Vision Transformer sourced from "https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html" for comparative purposes """
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_heads,
        num_layers,
        num_classes,
        num_patches,
        dropout=0.0,
    ):
        super().__init__()

        self.input_layer = nn.Linear(196, embed_dim)
        self.transformer = nn.Sequential(
            *(Other_AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        # Preprocess input
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out