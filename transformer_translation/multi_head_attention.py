import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention(nn.Module):
    """
    """

    def __init__(self, d_model, num_heads, d_queries, d_values):
        """
        :param d_model: the number of expected features in the input (the size of the input embeddings)
        :param d_queries: the number of features in the query and key vectors
        :param d_values: the number of features in the value vectors
        :param dropout_prob: the dropout probability
        :param apply_causal_mask: whether to apply a causal mask to the attention scores
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_queries = d_queries
        self.d_keys = d_queries
        self.d_values = d_values

        # Number of heads
        self.num_heads = num_heads

        # A linear linear to project the input query sequences to n_heads sets of queries
        self.project_queries = nn.Linear(d_model, self.num_heads * d_queries)

        # A linear layer to project the input key-value sequences to n_heads sets of keys-values
        self.project_keys_values = nn.Linear(d_model, self.num_heads * (self.d_keys + self.d_values))

        # A linear layer to project the concatenated attention heads back to d_model
        self.project_output = nn.Linear(self.num_heads * d_values, d_model)

        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_sequence, key_value_sequence, key_value_padding_mask = None, apply_causal_mask = False):
        """

        """

        # Query Sequence length (T_q)
        T_q = query_sequence.size(1)

        # Key-value sequence length (T_kv)
        T_kv = key_value_sequence.size(1)

        # Get batch size (B)
        B = query_sequence.size(0)

        # Project the input query sequences to n_heads sets of queries:
        proj_q = self.project_queries(query_sequence)                   # [B, T_q, n_heads * d_queries]

        # Project the input key-value sequences to n_heads sets of keys-values:
        proj_kv = self.project_keys_values(key_value_sequence)          # [B, T_kv, n_heads * (d_keys + d_values)]

        # Split into keys and values:
        # [B, T, n_heads* d_keys], [B, T, n_heads * d_values]
        proj_k, proj_v = proj_kv.split(split_size = self.num_heads * self.d_keys, dim = -1)

        # Compute the attention by performing a batch matrix multiplication of projected queries and projected keys

        # First split the last dimension to extract the individual queries, keys, values for each head:
        proj_q = proj_q.contiguous().view(B, T_q, self.num_heads, self.d_queries)
        proj_k = proj_k.contiguous().view(B, T_kv, self.num_heads, self.d_keys)
        proj_v = proj_v.contiguous().view(B, T_kv, self.num_heads, self.d_values)

        # Second permute the projection matrices so that the last two dimensions are [seq_length, d_{q/k/v}]
        # Merge the remaining dimensions so that each of them are 3D tensors
        # This is to prepare the projected queries, keys and values for batch matrix multiplication
        proj_q = proj_q.permute(0, 2, 1, 3).contiguous().view(-1, T_q, self.d_queries)   # [B*n_heads, T_q, d_queries]
        proj_k = proj_k.permute(0, 2, 1, 3).contiguous().view(-1, T_kv, self.d_keys)     # [B*n_heads, T_kv, d_keys]
        proj_v = proj_v.permute(0, 2, 1, 3).contiguous().view(-1, T_kv, self.d_values)   # [B*n_heads, T_kv, d_values]

        # Now actually perform the attention weights computation using batch matrix multiplication
        attention_weights = torch.bmm( proj_q, proj_k.permute(0, 2, 1))     # [B*n_heads, T_q, T_kv]

        # Scale the attention weights:
        attention_weights = (1./math.sqrt(self.d_keys)) * attention_weights

        # PERFORM MASKING ON ATTENTION WEIGHTS:
        # 1. Key value padding masking (Shape of key-value padding mask is: [B, T_kv] )
        attention_weights = attention_weights.contiguous().view(B, self.num_heads, T_q, T_kv).permute(1, 2, 0, 3)  # [n_heads, T_q, B, T_kv]
        attention_weights = attention_weights.masked_fill(key_value_padding_mask, -float('inf'))                   # [n_heads, T_q, B, T_kv]
        # Bring back to original shape:
        attention_weights = attention_weights.permute(2, 0, 1, 3).contiguous().view(-1, T_q, T_kv)

        # 2. Casual masking: 
        if apply_causal_mask == True:
            # torch.tril(), i.e. lower triangle in a 2D matrix, sets j > i to 0
            not_future_mask = torch.ones_like(attention_weights).tril().bool().to(device)  # (B * n_heads, T_q  , T_kv)

            # Mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
            attention_weights = attention_weights.masked_fill(~not_future_mask, -float('inf'))  # (B * n_heads, T_q, T_kv)

        # Apply softmax to the attention weights:
        attention_weights = self.softmax(attention_weights)   # [B*n_heads, T_q, T_kv]

        # For each query, sum up the values based on the computed and normalized attention weights. 
        # To optimize this, we again use the Batched Matrix Multiplication
        output_sequence = torch.bmm(attention_weights, proj_v)   # [B*n_heads, T_q, d_values]

        # Unmerge the batch size and num heads dimensions, and restore the original order
        output_sequence = output_sequence.contiguous().view(B, self.num_heads, T_q, self.d_values).permute(0, 2, 1, 3)  # [B, T_q, n_heads, d_values]

        # Merge the last two dimensions (n_heads and d_values)
        output_sequence = output_sequence.contiguous().view(B, T_q, -1)   # [B, T_q, n_heads * d_values]

        # Project the concatenated attention heads back to the model dimension:
        output_sequence = self.project_output(output_sequence)    # [B, T_q, d_model]

        return output_sequence



