import torch.nn as nn
import torch
import math

from encoder import Encoder
from decoder import Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Transformer(nn.Module):
    """
    The Transformer network.
    """

    def __init__(self, vocab_size, positional_encoding, num_layers = 6, d_model=512, n_heads=8, d_hidden=2048, d_queries=64, d_values=64, dropout_prob=0.1):
        """
        :param vocab_size: size of the (shared) vocabulary
        :param positional_encoding: positional encodings up to the maximum possible pad-length
        :param d_model: size of vectors throughout the transformer model
        :param n_heads: number of heads in the multi-head attention
        :param d_queries: size of query vectors (and also the size of the key vectors) in the multi-head attention
        :param d_values: size of value vectors in the multi-head attention
        :param d_inner: an intermediate size in the position-wise FC
        :param n_layers: number of layers in the Encoder and Decoder
        :param dropout: dropout probability
        """
        super(Transformer, self).__init__()

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # Encoder
        self.encoder = Encoder(vocab_size=vocab_size,
                               positional_encoding=positional_encoding,
                               num_layers = num_layers,
                               d_model=d_model,
                               n_heads=n_heads,
                               dropout_prob=dropout_prob,
                               d_hidden=d_hidden,
                               d_queries=d_queries,
                               d_values=d_values)

        # Decoder
        self.decoder = Decoder(vocab_size=vocab_size,
                               positional_encoding=positional_encoding,
                               num_layers = num_layers,
                               d_model=d_model,
                               n_heads=n_heads,
                               dropout_prob=dropout_prob,
                               d_hidden=d_hidden,
                               d_queries=d_queries,
                               d_values=d_values)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights in the transformer model.
        """
        # Glorot uniform initialization with a gain of 1.
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the tensor
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.)

        # Share weights between the embedding layers and the logit layer
        nn.init.normal_(self.encoder.token_embedding.weight, mean=0., std=math.pow(self.d_model, -0.5))
        self.decoder.token_embedding.weight = self.encoder.token_embedding.weight
        self.decoder.final_linear.weight = self.decoder.token_embedding.weight

        print("Model initialized.")

    def forward(self, encoder_sequences, decoder_sequences, encoder_padding_masks, decoder_padding_masks):
        """
        Forward propagation.

        :param encoder_sequences: source language sequences, a tensor of size (N, encoder_sequence_pad_length)
        :param decoder_sequences: target language sequences, a tensor of size (N, decoder_sequence_pad_length)
        :param encoder_padding_masks: 
        :param decoder_padding_masks: 
        :return: decoded target language sequences, a tensor of size (N, decoder_sequence_pad_length, vocab_size)
        """

        ### Encoder
        encoder_sequences = self.encoder(encoder_sequences,
                                         encoder_padding_masks)  # (N, encoder_sequence_pad_length, d_model)

        # Decoder
        decoder_sequences = self.decoder(decoder_sequences, decoder_padding_masks, encoder_sequences,
                                         encoder_padding_masks)  # (N, decoder_sequence_pad_length, vocab_size)

        return decoder_sequences