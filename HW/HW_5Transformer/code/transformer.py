import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=512, N=1_000.0):
        """
        Initialize the PositionalEncoding object.
        This will precompute the positional encoding for all possible sequences of length max_len.
        At runtime only the encoding for the input sequence length will be used.

        Parameters:
        embed_size: int
            The number of features in the input
        max_len: int
            The maximum length of the input sequence. This should be the maximum length of the input sequence you will ever use. If you have a an initial prompt length of 32 and want to generate 1000 characters, then max_len should be 1032.
        N: float
            A constant used in the positional encoding to control the scale of the frequencies.
            The slowest frequency will have period of N^((embed_size-2)/embed_size) * 2 * np.pi.
            This period is a little shorter than N * 2 * np.pi. N controls the longest positional encoding that can be identified.
            Any sequence longer than this will have the same positional encoding as a shorter sequence.
            So N/(2 * pi) should be longer than the longest sequence you will use.
        """
        super(PositionalEncoding, self).__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        pe = torch.zeros(max_len, embed_size)  # positional encoding
        # TODO create the positional encoding by filling in the values for pe
        pe = pe.unsqueeze(1)  # reshape to (max_len, 1, embed_size) so that it applies to each sequence in the mini-batch
        self.register_buffer('pe', pe)  # register the positional encoding as buffer so it is saved in the model state_dict but not listed as parameters

    def forward(self, x):
        """
        Forward pass of the PositionalEncoding.

        Parameters:
        x: torch.Tensor of shape (seq_len, batch_size, d_model)
            The input features

        Returns:
        torch.Tensor of shape (seq_len, batch_size, d_model)
            The input features with positional encoding added
        """
        return x + self.pe[:x.size(0)]

class TransformerLayer(nn.Module):
    def __init__(self, embed_size, nhead, layer_width=256, activation=F.relu):
        """
        Initialize the Transformer object.

        Parameters:
        embed_size: int
            The number of features in the input
        nhead: int
            The number of heads in the multiheadattention models
        layer_width: int
            The number of features in the feedforward network
        activation: callable function
            The activation function for the hidden layer in the feedforward network

        """
        super(TransformerLayer, self).__init__()
        self.embed_size = embed_size
        self.nhead = nhead
        # TODO create the multihead attention, layer normalization, feedforward network, and dropout layers

    def forward(self, x, attn_mask, need_attn_weights=False):
        """
        Forward pass of the Transformer.

        Parameters:
        x: torch.Tensor of shape (seq_len, batch_size, d_model)
            The input features

        attn_mask: torch.Tensor of shape (seq_len, seq_len)
            The mask to apply to the attention weights

        Returns:
        torch.Tensor of shape (seq_len, batch_size, d_model)
            The output features
        """

        # TODO fill in the steps below
        # normalize input
        xnorm = None

        # self attention using normalized input. 
        # Note that the query, key, and value inputs are all the same here when doing self-attention with nn.MultiheadAttention. These inputs are used to generate the query, key, and value vectors and are not the keys themselves.
        # Make sure to pass in the attention mask, set is_causal=True, set need_weights=need_attn_weights, and set average_attn_weights=False
        att, att_weights = None, None
        
        # add attention to original input x
        y = None
        
        # layer norm and feedforward 
        z = None
        
        # apply dropout to the output of the feedforward network
        z = None
        
        # add feedforward output to z to y to get the final output
        z = None

        return z, att_weights

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size=32, num_layers=2, nhead=8, layer_width=256, activation=F.relu, max_len=1024+32, N=1_000.0, pos_enc=True):
        """
        Initialize the Transformer object.

        Parameters:
        vocab_size: int
            The size of the vocabulary
        embed_size: int
            The number of features for the embedding layer
        num_layers: int
            The number of transformer layers
        nhead: int
            The number of heads in the multiheadattention models
        layer_width: int
            The number of features in the feedforward network
        activation: callable function
            The activation function for the hidden layer in the feedforward network
        max_len: int
            The maximum length of the input sequence. This is passed to the PositionalEncoding object.
        N: float
            Constant used in the PositionalEncoding object to specify the the slowest frequency.
        pos_enc: bool
            Flag to use positional encoding or not
        """
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if pos_enc:
            self.position = PositionalEncoding(embed_size, max_len=max_len, N=N)
        else:
            self.position = nn.Identity()  # identity function does nothing
        self.transformer_layers = nn.ModuleList([None])  # TODO replace with tranformer layers
        # TODO add layer norm and final linear layer

    def forward(self, x, attn_mask, return_att_weights=False):
        """
        Forward pass of the Transformer.

        Parameters:
        x: torch.Tensor of shape (seq_len, batch_size)
            The input features

        mask: torch.Tensor of shape (seq_len, seq_len)
            The mask to apply to the attention weights

        return_att_weights: bool
            Whether to return the attention weights or not. If False, the function will not record the attention weights. 

        Returns:
        torch.Tensor of shape (seq_len, batch_size, vocab_size)
            The output features
        """
        x = self.embedding(x)
        x = self.position(x)  # this already does the addition of positional encoding with the embedding
        att_weights = []
        for layer in self.transformer_layers:
            x, att_w = layer(x, attn_mask=attn_mask, need_attn_weights=return_att_weights)
            if return_att_weights:
                att_weights.append(att_w)
        
        x = None  # TODO normalize the output of the last transformer layer and pass it through the final linear layer
        if return_att_weights:
            return x, att_weights
        return x

