import torch

class FullyConnected(torch.nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()

        input_size = 8
        hidden_layer_size = 64
        hidden_layer_2_size = 128
        hidden_layer_3_size = 32
        output_size = 1

        self.fc1 = torch.nn.Linear(input_size, output_size)
        # self.fc2 = torch.nn.Linear(hidden_layer_size, hidden_layer_2_size)
        # self.fc3 = torch.nn.Linear(hidden_layer_2_size, hidden_layer_3_size)
        # self.fc4 = torch.nn.Linear(hidden_layer_3_size, output_size)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        # x = self.fc4(x)
        x = self.sigmoid(x)
        return x