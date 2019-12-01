import torch

class FullyConnected(torch.nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()

        input_size = 8
        hidden_layer_size = 128
        hidden_layer_2_size = 64
        output_size = 2

        self.fc1 = torch.nn.Linear(14 * 4 * 23, hidden_layer_size)
        self.fc2 = torch.nn.Linear(hidden_layer_size, hidden_layer_2_size)
        self.fc3 = torch.nn.Linear(hidden_layer_2_size, output_size)
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x