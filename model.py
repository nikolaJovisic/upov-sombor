import torch
import torch.nn as nn


class WaterNet(nn.Module):
    def __init__(
        self, features_size, seq_length, seq_channels=2, output_size=1, mode="mlp"
    ):
        """
        :param features_size: Size of inputs passed directly.
        :param seq_length: Window size for sampling.
        :param seq_channels: Number of features (channels) in each element of the sequence
        (typically 2 because _ulaz and _izlaz are used)
        :param output_size: Number of predicted features (typically only 1)
        :param mode: Sequence part of the model mode - 'mlp', 'lstm' or 'conv'
        """
        super(WaterNet, self).__init__()
        self.mode = mode
        neurons = 200

        if self.mode == "lstm":
            lstm_hidden_size = 10
            self.lstm = nn.LSTM(
                input_size=output_size * seq_channels,
                hidden_size=lstm_hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.input_fc = nn.Linear(features_size + lstm_hidden_size, neurons)
        elif self.mode == "conv":
            conv_out_channels = 3
            conv_kernel_size = 5
            self.conv1d = nn.Conv1d(
                in_channels=output_size * seq_channels,
                out_channels=conv_out_channels,
                kernel_size=conv_kernel_size,
            )
            self.input_fc = nn.Linear(
                features_size + conv_out_channels * (seq_length - conv_kernel_size + 1),
                neurons,
            )
        else:
            self.input_fc = nn.Linear(features_size, neurons)

        self.output_fc = nn.Linear(neurons, output_size)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        if self.mode == "lstm":
            feature_inputs, lstm_inputs = inputs
            lstm_output, _ = self.lstm(lstm_inputs)
            lstm_output = lstm_output[:, -1, :]
            x = torch.cat((lstm_output, feature_inputs), dim=1)
        elif self.mode == "conv":
            feature_inputs, conv_inputs = inputs
            conv_output = self.conv1d(conv_inputs.permute(0, 2, 1))
            conv_output = torch.cat(torch.unbind(conv_output, dim=1), dim=1)
            x = torch.cat((conv_output, feature_inputs), dim=1)
        else:
            x = inputs

        x = self.input_fc(x)
        x = self.relu(x)
        x = self.output_fc(x)
        return x
