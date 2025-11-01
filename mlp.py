import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):

    def __init__(self,
                d_input: int = 2,
                d_hidden: int = 32,
                d_output: int = 1,
                activation = 'tanh'):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(d_input, d_hidden)
        self.linear2 = torch.nn.Linear(d_hidden, d_hidden)
        self.linear3 = torch.nn.Linear(d_hidden, d_output)

        self.activation_fn = self.get_activation_fn(activation)

    def forward(self, z):
        z = self.activation_fn(self.linear1(z))
        z = self.activation_fn(self.linear2(z))
        return self.linear3(z)

    def get_activation_fn(self, activation):
        activation = activation.lower()
        if activation == 'tanh':
            act =  torch.tanh
        elif activation == 'relu':
            act = torch.relu
        elif activation == 'sigmoid':
            act = torch.sigmoid
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        return act
        

        