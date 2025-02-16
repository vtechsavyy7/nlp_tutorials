import torch.nn as nn

## Explanation: 
# The input to the encoder is a sequence of indices where  each index corresponds to a word in the source language
# The embedding layer is a simple mapping from a given index to a vector of floats representing the word. The dimension
# of the vector representing the words is a param of the embedding layer

# After the embedding layer, the sequence of indices is basically transformed to a sequence of vectors going into the GRU layer

# The outputs from the encoder are the last output state and last hidden state of the GRU layer


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden