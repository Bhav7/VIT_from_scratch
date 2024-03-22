import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """ Implementation of Vaswani et als. Multi-Head Attention Layer"""
    def __init__(self, num_heads, qk_embedding_dim, v_embedding_dim):
        super().__init__()
        self.num_heads = num_heads
        self.model_embedding_dim = v_embedding_dim

        self.QW = nn.Linear(qk_embedding_dim, qk_embedding_dim)
        self.KW = nn.Linear(qk_embedding_dim, qk_embedding_dim)
        self.VW = nn.Linear(v_embedding_dim, v_embedding_dim)

        self.qk_embedding_dim_per_head = int(qk_embedding_dim/num_heads)
        self.v_embedding_dim_per_head = int(v_embedding_dim/num_heads)


        self.output_layer = nn.Linear(v_embedding_dim, self.model_embedding_dim)

    def forward(self, input):
        #Input shape: batch_size x num_tokens x  og_qk_embedding_dimensions
        b = input.shape[0]

        query = self.QW(input)
        #query shape: batch_size x num_tokens x og_qk_embedding_dimensions; same for key and value when post-linear transformation
        query = query.view(b, -1 , self.num_heads, self.qk_embedding_dim_per_head).permute(0,2,1,3)
        #query shape: batch_size x num_heads x num_tokens x og_qk_embedding_dimensions/num_heads

        key = self.KW(input)
        key = key.view(b, -1 , self.num_heads, self.qk_embedding_dim_per_head).permute(0,2,1,3)
        #key shape: batch_size x num_heads x num_tokens x og_qk_embedding_dimensions/num_heads

        value = self.VW(input)
        value = value.view(b, -1 , self.num_heads, self.v_embedding_dim_per_head).permute(0,2,1,3)
        #value shape: batch_size x num_heads x num_tokens x og_v_embedding_dimensions/num_heads

        x = (torch.matmul(query, key.permute(0,1,3,2)))/(self.qk_embedding_dim_per_head**(1/2))
        #key.permute(0,1,3,2) shape: batch_size x num_heads x og_v_embedding_dimensions/num_heads x num_tokens
        #x shape: batch_size x num_heads x num_tokens x num_tokens -> unnormalized attention matrix
        x = F.softmax(x, dim = -1)

        output = torch.matmul(x, value)
        #output shape: batch_size x num_head x num_tokens x og_v_embedding_dimensions/num_heads

        output = output.permute(0,2,1,3).reshape(b, -1, self.num_heads * self.v_embedding_dim_per_head)
        #output shape: batch_size x num_tokens x og_v_embedding_dimensions
        output = self.output_layer(output)
        #output shape: batch_size x num_tokens x og_v_embedding_dimensions
        return output


class EncoderBlock(nn.Module):
    """ Implementation of Dosovitskiy et als. VIT Encoder Block"""
    def __init__(self, heads, kq_embedding_dimensions, v_embedding_dimensions):
        super().__init__()
        self.attention_sublayer = MultiHeadAttention(heads, kq_embedding_dimensions, v_embedding_dimensions)
        self.layer_norm_1 = nn.LayerNorm(v_embedding_dimensions)
        self.ffn_sublayer = nn.Sequential(nn.Linear(v_embedding_dimensions, v_embedding_dimensions*4), nn.GELU(), nn.Linear(v_embedding_dimensions*4, v_embedding_dimensions))
        self.layer_norm_2 = nn.LayerNorm(v_embedding_dimensions)
    
    def forward(self, input):
        #Input shape: batch_size x num_tokens x qk_embedding_dimensions
        x_sub = self.layer_norm_1(input)
        x_sub = self.attention_sublayer(x_sub)
        x = x_sub + input
        #x shape: batch_size x num_tokens x  v_embedding_dimensions; Note: qk_embedding_dimensions == v_embedding_dimensions
        x_sub = self.layer_norm_2(x)
        x_sub = self.ffn_sublayer(x_sub)
        output = x_sub + x
        #output shape: batch_size x num_tokens x  v_embedding_dimensions
        return output
    

class Other_AttentionBlock(nn.Module):
    """ Attention Block sourced from "https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html" for comparative purposes """
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x