import torch.nn as nn


class WaterNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WaterNet, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_fc(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.output_fc(x)
        return x
