import torch 
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    """
    A neural network with one hidden layer. Uses the leaky RELU activation function.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def init_weights(self):
        init.xavier_normal(self.fc1.weight) 
        init.xavier_normal(self.fc2.weight)
        
    def forward(self, x_in, apply_softmax=False):
        a_1 = F.leaky_relu(self.fc1(x_in))
        y_pred = F.leaky_relu(self.fc2(a_1))
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred

    def predict(self, x):
        return self.forward(torch.tensor(x.values).float())