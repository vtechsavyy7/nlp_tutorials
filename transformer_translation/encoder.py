import torch
import torch.nn as nn
import math

from encoder_block import EncoderBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    """

    def __init__(self, vocab_size, positional_encoding, num_layers, d_model, n_heads, dropout_prob, d_hidden, d_queries, d_values):
        
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.positional_encoding = positional_encoding

        # Token embedding layer:
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Encoder layers:
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, dropout_prob, d_hidden, d_queries, d_values) for i in range(num_layers)])

    def forward(self, input_sequence, input_padding_mask):
        """
        """

        # Get the padded length: 
        padded_length = input_sequence.size(1)  # pad-length of this batch only, varies across batches

        # Apply positional encoding: 
        # Sum vocab embeddings and position embeddings
        input_sequence = self.token_embedding(input_sequence) * math.sqrt(self.d_model) + self.positional_encoding[:, :padded_length, :].to(device)  # (B, padded_length, d_model)

        # Apply the Encoder blocks sequentially
        for encoder_block in self.encoder_blocks:

            input_sequence = encoder_block(input_sequence, input_padding_mask)

        return input_sequence