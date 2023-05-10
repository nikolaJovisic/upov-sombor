import torch
import torch.nn as nn


class WaterNet(nn.Module):
    def __init__(self, features_size, lstm_input_size, output_size, use_lstm=False):
        super(WaterNet, self).__init__()
        self.use_lstm = use_lstm
        neurons = 100
        if self.use_lstm:
            lstm_hidden_size = 10
            self.lstm = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=lstm_hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.input_fc = nn.Linear(features_size + lstm_hidden_size, neurons)
        else:
            self.input_fc = nn.Linear(features_size, neurons)
        self.output_fc = nn.Linear(neurons, output_size)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        if self.use_lstm:
            feature_inputs, lstm_inputs = inputs
            lstm_output, _ = self.lstm(lstm_inputs)
            lstm_output = lstm_output[:, -1, :]
            x = torch.cat((lstm_output, feature_inputs), dim=1)
        else:
            x = inputs
        x = self.input_fc(x)
        x = self.relu(x)
        x = self.output_fc(x)
        return x
