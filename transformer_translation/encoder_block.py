import torch 
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from pffn import PFFN

class EncoderBlock(nn.Module):
    """
    """

    def __init__(self, d_model, num_heads,  dropout_prob, d_hidden, d_queries, d_values):
        super().__init__()

        # Multi-head attention layer
        self.mha = MultiHeadAttention(d_model, num_heads, d_queries, d_values)

        # Position-wise feed forward layer
        self.pffn = PFFN(d_model, d_hidden)

        # First dropout layer: 
        self.apply_dropout_1 = nn.Dropout(dropout_prob)

        # Second dropout layer: 
        self.apply_dropout_2 = nn.Dropout(dropout_prob)

        # First layer normalization: 
        self.ln_1 = nn.LayerNorm(d_model)

        # Second layer normalization: 
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, input_sequence, input_padding_mask):
        """
        """

        ### Part 1: Multi-Head Attention:
        # Apply multi-head attention ((Don't apply the causal mask))
        mha_output_sequence = self.mha(input_sequence, input_sequence, input_padding_mask, apply_causal_mask=False)

        # Apply dropout: 
        mha_output_sequence = self.apply_dropout_1(mha_output_sequence)

        # Add residual connection: 
        mha_output_sequence = mha_output_sequence + input_sequence

        # Apply layer normalization
        mha_output_sequence = self.ln_1(mha_output_sequence)

        ### Part 2: Position-wise Feed-forward network:
        # Apply the PFFN: 
        pffn_output_sequence = self.pffn(mha_output_sequence)

        # Apply dropout: 
        pffn_output_sequence = self.apply_dropout_2(pffn_output_sequence)

        # Add the residual connection: 
        pffn_output_sequence = pffn_output_sequence + mha_output_sequence

        # Apply layer normalization: 
        pffn_output_sequence = self.ln_2(pffn_output_sequence)

        return pffn_output_sequence



