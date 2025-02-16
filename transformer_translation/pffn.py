import torch 
import torch.nn as nn


class PFFN(nn.Module):
    """
    
    """

    def __init__(self, d_model, d_hidden):
        super().__init__()

        # First Linear layer: 
        self.linear_1 = nn.Linear(d_model, d_hidden)

        # ReLU layer: 
        self.relu = nn.ReLU()

        # Second linear layer: 
        self.linear_2 = nn.Linear(d_hidden, d_model)

    def forward(self, sequence):
        """
        
        """

        # Apply the forward operation; 
        output_sequence = self.linear_2(self.relu(self.linear_1(sequence)))

        return output_sequence