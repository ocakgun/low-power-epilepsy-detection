import torch.utils.data as data
from torch.utils.data import Dataset
from pathlib import Path
import pyedflib
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class EpilepsyData(Dataset):
    def __init__(self, data):
        # Get directory listing from path

        self.items = data
        self.length = len(self.items)

    def __getitem__(self, index):
        filename, label = self.items[index]
        signals = pyedflib.highlevel.read_edf(filename)
        return signals, int(label)

    def __len__(self):
        return self.length


bs = 12
EEG_DATA = "/home/jmsvanrijn/Documents/Afstuderen/Code/low-power-epilepsy-detection/data/processed/"

# Create a array seizures, 1= seizure, 0= normal
seizures_2 = sorted(Path(EEG_DATA).glob('seizures/*.edf'))
seizure_files_2 = [(str(seizure), str(1)) for seizure in seizures_2]

normals_2 = sorted(Path(EEG_DATA).glob('normal/*.edf'))
normal_files_2 = [(str(normal), str(0)) for normal in normals_2]

all_data = seizure_files_2 + normal_files_2

# Randomly shuffle data
random.shuffle(all_data)

# Select randomly
val_percentage = 0.2  # 20%
train_percentage = 0.8  # 80%

val_length = round(len(all_data)*val_percentage)
train_length = round(len(all_data)*train_percentage)

val_dataset = EpilepsyData(all_data[:val_length])
train_dataset = EpilepsyData(all_data[val_length:])

val_loader = data.DataLoader(val_dataset, batch_size=bs, shuffle=True)
train_loader = data.DataLoader(train_dataset, batch_size=bs, shuffle=True)

print("Success")


class EpilepsyNet(nn.Module):
    def __init__(self):
        super(EpilepsyNet, self).__init__()
        self.conv1 = nn.Conv1d(100, 128, kernel_size=5, stride=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        # self.pool4 = nn.MaxPool1d(4)
        self.pool4 = nn.MaxPool1d(1)
        # self.avgPool = nn.AvgPool1d(30)
        self.avgPool = nn.AvgPool1d(4)
        self.fc1 = nn.Linear(512, 50)

    def forward(self, x):
        x = x.unsqueeze(-1).view(-1, 100, 2205)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)
        return x


epilepsy_net = EpilepsyNet()

if torch.cuda.is_available():
    epilepsy_net.to("cuda")


def train_2(model, optimizer, train_loader):
    print(model)
    print(optimizer)


lr = 0.001  # 0.21
optimizer = optim.Adam(epilepsy_net.parameters(), lr=lr)

train_2(epilepsy_net, optimizer, train_loader)

# def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
#     for epoch in range(epochs):
#         training_loss = 0.0
#         valid_loss = 0.0
#         model.train()
#         for batch in train_loader:
#             optimizer.zero_grad()
#             inputs, targets = batch
#             inputs = inputs.to(device)
#             targets = targets.to(device)
#             output = model(inputs)
#             loss = loss_fn(output, targets)
#             loss.backward()
#             optimizer.step()
#             training_loss += loss.data.item() * inputs.size(0)
#         training_loss /= len(train_loader.dataset)
#
#         model.eval()
#         num_correct = 0
#         num_examples = 0
#         for batch in val_loader:
#             inputs, targets = batch
#             inputs = inputs.to(device)
#             output = model(inputs)
#             targets = targets.to(device)
#             loss = loss_fn(output, targets)
#             valid_loss += loss.data.item() * inputs.size(0)
#             correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
#             num_correct += torch.sum(correct).item()
#             num_examples += correct.shape[0]
#         valid_loss /= len(val_loader.dataset)
#
#         print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss, valid_loss, num_correct / num_examples))


# lr = 0.001  # 0.21
# optimizer = optim.Adam(epilepsy_net.parameters(), lr=lr)
# print("Here")
# train(epilepsy_net, optimizer, torch.nn.CrossEntropyLoss(), train_loader, val_loader, epochs=20, device="cpu")
