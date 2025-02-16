import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import *

## Explanation: 

# The main input to the decoder is the last hidden state from the encoder
# The input state of the decoder is initalized to the SOS_token

# The deocder has its own embedding layer from which it looks up target language word embeddings based on word index
# The main layer of the decoder is a GRU. 
# The GRU layer is followed by a linear layer that maps from the GRU output to a vector of scores for each word in the target language
# The GRU is not run in a conventional manner. Its running operation depends
# on whether we're in training or inference mode. In training mode, we use 'Teacher-Forcing' which means that the input to the next 
# step of the decoder is the word from the ground truth target sequence
# In inference, we use the most proabable previous output for the next step input to the decoder


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden