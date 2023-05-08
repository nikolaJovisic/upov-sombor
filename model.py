import torch
import torch.nn as nn


class WaterNet(nn.Module):
    def __init__(self, features_size, lstm_input_size, output_size):
        super(WaterNet, self).__init__()
        hidden_size = 10
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, hidden_size=hidden_size, num_layers=1, batch_first=True
        )
        self.input_fc = nn.Linear(features_size + hidden_size, 200)
        self.output_fc = nn.Linear(200, output_size)
        self.relu = nn.ReLU()

    def forward(self, feature_inputs, lstm_inputs):
        lstm_output, _ = self.lstm(lstm_inputs)
        lstm_output = lstm_output[:,-1,:]
        x = torch.cat((lstm_output, feature_inputs), dim=1)
        x = self.input_fc(x)
        x = self.relu(x)
        x = self.output_fc(x)
        return x
