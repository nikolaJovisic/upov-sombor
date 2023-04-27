import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import INPUT_COLUMNS, OUTPUT_COLUMNS, WaterDataset
from model import WaterNet

dataset = WaterDataset()

train_set, validation_set, test_set = random_split(dataset, lengths=(0.7, 0.2, 0.1))

batch_size = 8

train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=True
)
validation_loader = DataLoader(
    validation_set, batch_size=batch_size, shuffle=True, drop_last=True
)

input_dim = len(INPUT_COLUMNS)
output_dim = len(OUTPUT_COLUMNS)
hidden_dim = 150

model = WaterNet(input_dim, hidden_dim, output_dim)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

best_validation_loss = np.inf

train_loss_history = []
validation_loss_history = []
train_r2_history = []
validation_r2_history = []

for epoch in range(5000):
    model.train()
    train_loss = 0.0
    train_outputs = []
    train_labels = []
    for data in train_loader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_outputs.append(outputs)
        train_labels.append(labels)

    train_loss /= len(train_loader)
    with torch.no_grad():
        train_outputs = torch.cat(train_outputs).numpy()
        train_labels = torch.cat(train_labels).numpy()
        train_r2 = r2_score(train_labels, train_outputs)

    validation_loss = 0.0
    validation_outputs = []
    validation_labels = []
    with torch.no_grad():
        for data in validation_loader:
            inputs, labels = data
            outputs = model(inputs)
            validation_loss += criterion(outputs, labels).item()
            validation_outputs.append(outputs)
            validation_labels.append(labels)

    validation_loss /= len(validation_loader)
    with torch.no_grad():
        validation_outputs = torch.cat(validation_outputs).numpy()
        validation_labels = torch.cat(validation_labels).numpy()
        validation_r2 = r2_score(validation_labels, validation_outputs)

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
        f"validation_r2:{validation_r2:.3f}"
    )

    train_loss_history.append(train_loss)
    validation_loss_history.append(validation_loss)
    train_r2_history.append(train_r2)
    validation_r2_history.append(validation_r2)

    # plt.plot(train_loss_history, label="train_loss")
    # plt.plot(validation_loss_history, label="validation_loss")
    # plt.legend()
    # plt.show()

print("Finished Training")
