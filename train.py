import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from torch import nn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error as mape
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset

from dataset import FEATURE_COLUMNS, OUTPUT_COLUMNS, SEQUENCE_LENGTH, WaterDataset, SEQUENCE_CHANNELS, load, preprocess
from model import WaterNet

def rmse(*args):
    return np.sqrt(mean_squared_error(*args))


mode = "mlp"
use_seq = mode != "mlp"

data = load()
train_valid_df, test_df = preprocess(data, use_seq)

train_valid_set = WaterDataset(train_valid_df, use_seq=use_seq)
test_set = WaterDataset(test_df, use_seq=use_seq)

train_size = int(0.8 * len(train_valid_set))

train_set = Subset(train_valid_set, range(train_size))
validation_set = Subset(train_valid_set, range(train_size + SEQUENCE_LENGTH, len(train_valid_set)))


batch_size = 8

train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=False, drop_last=True
)
validation_loader = DataLoader(
    validation_set, batch_size=batch_size, shuffle=False, drop_last=True
)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

feature_size = len(FEATURE_COLUMNS)
output_size = len(OUTPUT_COLUMNS)

model = WaterNet(
    features_size=feature_size,
    seq_length=SEQUENCE_LENGTH,
    seq_channels=len(SEQUENCE_CHANNELS),
    output_size=output_size,
    mode=mode,
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = ExponentialLR(optimizer, gamma=0.9)
best_validation_loss = np.inf

train_loss_history = []
validation_loss_history = []
train_r2_history = []
validation_r2_history = []


for epoch in range(30):
    model.train()
    train_loss = 0.0
    train_outputs = []
    train_labels = []
    for data in train_loader:
        input, labels = data
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_outputs.append(outputs)
        train_labels.append(labels)

    scheduler.step()
    train_loss /= len(train_loader)

    model.eval()

    with torch.no_grad():
        train_outputs = torch.cat(train_outputs)
        train_labels = torch.cat(train_labels)
        train_r2 = r2_score(train_labels.numpy(), train_outputs.numpy())
        train_rmse = rmse(train_labels, train_outputs)
        train_mape = mape(train_labels, train_outputs)

        validation_loss = 0.0
        validation_outputs = []
        validation_labels = []

        for data in validation_loader:
            input, labels = data
            outputs = model(input)
            validation_loss += criterion(outputs, labels).item()
            validation_outputs.append(outputs)
            validation_labels.append(labels)

        validation_loss /= len(validation_loader)

        validation_outputs = torch.cat(validation_outputs)
        validation_labels = torch.cat(validation_labels)
        validation_r2 = r2_score(validation_labels.numpy(), validation_outputs.numpy())
        validation_rmse = rmse(validation_labels, validation_outputs)
        validation_mape = mape(validation_labels, validation_outputs)

        if validation_loss < best_validation_loss:
            print(
                f"New best model, loss update: {best_validation_loss} -> {validation_loss}"
            )
            best_validation_loss = validation_loss
            torch.save(
                model.state_dict(),
                "water_net.pt",
            )

        print(
            f"Epoch: [{epoch + 1}], "
            f"train_loss:{train_loss:.3f}, "
            f"validation_loss:{validation_loss:.3f}, "
            f"train_r2:{train_r2:.3f}, "
            f"validation_r2:{validation_r2:.3f}, "
            f"train_rmse:{train_rmse:.3f}, "
            f"validation_rmse:{validation_rmse:.3f}, "
            f"train_mape:{train_mape:.3f}, "
            f"validation_mape:{validation_mape:.3f}"
        )

        train_loss_history.append(train_loss)
        validation_loss_history.append(validation_loss)
        train_r2_history.append(train_r2)
        validation_r2_history.append(validation_r2)

model.load_state_dict(torch.load("water_net.pt"))
model.eval()

with torch.no_grad():
    test_outputs = []
    test_labels = []
    for data in test_loader:
        input, labels = data
        outputs = model(input)
        test_outputs.append(outputs)
        test_labels.append(labels)
    test_outputs = torch.cat(test_outputs)
    test_labels = torch.cat(test_labels)
    test_r2 = r2_score(test_labels.numpy(), test_outputs.numpy())
    test_rmse = rmse(test_labels, test_outputs)
    test_mape = mape(test_labels, test_outputs)

    print(f"test_r2:{test_r2:.3f} test_rmse:{test_rmse:.3f}, test_mape:{test_mape:.3f}")
    # for i in zip(validation_labels, validation_outputs):
    #     print(i)

    num_outputs = test_outputs.size(1)
    for i in range(num_outputs):
        output = test_outputs[:, i].numpy()
        label = test_labels[:, i].numpy()

        # Creating a scatter plot
        plt.scatter(label, output)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("COD_" + mode)
        plt.show()
