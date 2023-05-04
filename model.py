import torch.nn as nn


class WaterNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WaterNet, self).__init__()
        self.input_fc = nn.Linear(input_dim, 50)
        self.fc1 = nn.Linear(50, 100)
        # self.fc2 = nn.Linear(100, 200)
        # self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 50)
        self.output_fc = nn.Linear(50, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_fc(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.output_fc(x)
        return x
