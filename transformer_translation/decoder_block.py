import torch 
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from pffn import PFFN

class DecoderBlock(nn.Module):
    """
    """

    def __init__(self, d_model, num_heads, dropout_prob, d_hidden, d_queries, d_values, ):
        super().__init__()

        # Multi-head self-attention layer 
        self.mha_self = MultiHeadAttention(d_model, num_heads, d_queries, d_values)

        # Multi-head cross-attention layer
        self.mha_cross = MultiHeadAttention(d_model, num_heads, d_queries, d_values)

        # Position-wise feed forward network
        self.pffn = PFFN(d_model, d_hidden)

        # First dropout layer: 
        self.apply_dropout_1 = nn.Dropout(dropout_prob)

        # Second dropout layer: 
        self.apply_dropout_2 = nn.Dropout(dropout_prob)

        # Third dropout layer: 
        self.apply_dropout_3 = nn.Dropout(dropout_prob)

        # First layer normalization: 
        self.ln_1 = nn.LayerNorm(d_model)

        # Second layer normalization: 
        self.ln_2 = nn.LayerNorm(d_model)

        # Third layer normalization: 
        self.ln_3 = nn.LayerNorm(d_model)

    def forward(self, input_sequence, input_padding_mask, memory_sequence, memory_padding_mask):
        """
        """

        ## Part 1: Multi-head self attention: 
        # Multi-head self attention (Apply the causal mask so that we don't look into the future while training)
        mha_self_output = self.mha_self(input_sequence, input_sequence, input_padding_mask, apply_causal_mask = True)

        # Apply dropout
        mha_self_output = self.apply_dropout_1(mha_self_output)

        # Add residual connection
        mha_self_output = mha_self_output + input_sequence

        # Apply layer normalization:
        mha_self_output = self.ln_1(mha_self_output)

        ## Part 2: Multi-head cross attention:
        # Multi-head cross attention (Don't apply the causal mask for cross-attention)
        mha_cross_output = self.mha_cross(mha_self_output, memory_sequence, memory_padding_mask, apply_causal_mask=False)

        # Apply dropout
        mha_cross_output = self.apply_dropout_2(mha_cross_output)

        # Add residual connection
        mha_cross_output = mha_cross_output + mha_self_output

        # Apply layer normalization:
        mha_cross_output = self.ln_2(mha_cross_output)

        ## Part 3: Position-wise feed forward network
        # PFFN: 
        pffn_output = self.pffn(mha_cross_output)

        # Apply dropout
        pffn_output = self.apply_dropout_2(pffn_output)

        # Add residual connection
        pffn_output = pffn_output + mha_cross_output

        # Apply layer normalization:
        pffn_output = self.ln_2(pffn_output)

        return pffn_output


