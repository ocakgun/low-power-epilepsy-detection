# convolutional network model we will train to detect patterns in readings. For more information see my tutorial here.
# Found at; https://github.com/SamLynnEvans/EEG-grasp-and-lift

from pathlib import Path
import random
import pyedflib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
data_loader = DataLoader(all_data, batch_size=bs, shuffle=True)


class convmodel(nn.Module):
    def __init__(self, out_classes, drop=0.5, d_linear=124):
        super().__init__()

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=0, stride=1)
        self.bn = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.linear1 = nn.Linear(8128, d_linear)

        self.linear3 = nn.Linear(d_linear, out_classes)
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)
        self.dropout3 = nn.Dropout(drop)

        self.conv = nn.Sequential(self.conv2, nn.ReLU(inplace=True), self.bn, self.pool, self.dropout1)
        self.dense = nn.Sequential(self.linear1, nn.ReLU(inplace=True), self.dropout2, self.dropout3, self.linear3)

    def forward(self, x):
        bs = x.size(0)
        x = self.conv(x)
        x = x.view(bs, -1)
        output = self.dense(x)

        return torch.sigmoid(output)


epilepsy_model = convmodel(2).double()


def get_samples(batch):
    inputs, labels = batch
    return inputs, labels


def train(data_loader, epochs, optimizer, model,printevery=100, shuffle=True, device="cpu"):
    epilepsy_model.train()
    for epochs in range(epochs):

        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            inputs, labels = get_samples(batch)
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            outputs = model(inputs)
            loss = F.binaray_cross_entryopy(outputs.view(-1), labels.view(-1))
            loss.backward()
            total_loss += loss.data[0]
            optimizer.step()


lr = 0.01
optimizer = optim.Adam(epilepsy_model.parameters(), lr=lr)
train(data_loader, 1, optimizer, epilepsy_model)