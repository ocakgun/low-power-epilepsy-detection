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
import numpy as np
import matplotlib.pyplot as plt


class EpilepsyData(Dataset):
    def __init__(self, data):
        # Get directory listing from path
        self.items = data
        self.length = len(self.items)

    def __getitem__(self, index):
        filename, label = self.items[index]
        signals, signal_headers, header = pyedflib.highlevel.read_edf(filename)

        loc = random.randint(0, len(signals[0]) - 1024)
        signals_cut = signals[:, loc: loc + 1024]
        return signals_cut, int(label)

    def __len__(self):
        return self.length


bs = 48
EEG_DATA = "/home/jmsvanrijn/Documents/Afstuderen/Code/low-power-epilepsy-detection/data/processed/"

# Create a array seizures, 1= seizure, 0= normal
seizures_2 = sorted(Path(EEG_DATA).glob('seizures/*.edf'))
seizure_files_2 = [(str(seizure), str(1)) for seizure in seizures_2]

normals_2 = sorted(Path(EEG_DATA).glob('normal/*.edf'))
normal_files_2 = [(str(normal), str(0)) for normal in normals_2]

all_data = seizure_files_2 + normal_files_2
traindata = EpilepsyData(all_data)
data_loader = DataLoader(traindata, batch_size=bs, shuffle=True)


class convmodel(nn.Module):
    def __init__(self, out_classes, drop=0.5, d_linear=124):
        super().__init__()

        self.conv2 = nn.Conv1d(23, 46, kernel_size=3, padding=0, stride=1)
        self.bn = nn.BatchNorm1d(46)
        self.pool = nn.MaxPool1d(2, stride=2)
        # self.linear1 = nn.Linear(5842, d_linear)
        self.linear1 = nn.Linear(511*46, d_linear)

        self.linear3 = nn.Linear(d_linear, out_classes)
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)
        self.dropout3 = nn.Dropout(drop)

        self.conv = nn.Sequential(self.conv2, nn.ReLU(inplace=True), self.bn, self.pool, self.dropout1)
        self.dense = nn.Sequential(self.linear1, nn.ReLU(inplace=True), self.dropout2, self.dropout3, self.linear3)

    def forward(self, x):
        bs = x.size(0)
        # print(x.size())
        x = self.conv(x)
        # print(x.size())
        x = x.view(bs, -1)
        # print(x.size())
        output = self.dense(x)

        return torch.sigmoid(output)


epilepsy_model = convmodel(1).double()


class SingleLayerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size, bias=False)
        self.relu = nn.ReLU()
        self.softmin = nn.Softmin(dim=-1)

    def forward(self, x):
        bs = x.size(0)
        out = x.view(bs, -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.softmin(out)
        return out


epilepsy_model_2 = SingleLayerNN(23*1024, 1).double()


def train(data_loader, epochs, optimizer, model, print_every=1, shuffle=True, device="cpu"):
    epilepsy_model.train()
    for epoch in range(epochs):
        total_loss = 0
        epoch_loss = []
        epoch_correct = []
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = F.binary_cross_entropy(outputs.view(-1), labels.double())
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
            # print(torch.eq(torch.round(outputs).view(-1), labels))
            correct = torch.sum(torch.eq(torch.round(outputs).view(-1), labels) == True)


            if (i + 1) % print_every == 0:
                print("epoch: {:d}, iter {:d}/{:d}, loss {:.4f}, Correct: {:.2f}%".format(
                epoch + 1, i + 1, len(traindata) // bs + 1, total_loss / print_every, correct.item()/len(inputs)*100))
                total_loss = 0

            epoch_loss.append(total_loss)
            epoch_correct.append(correct.item())

        print(sum(epoch_correct))
        print("Correct: {:.2f}%".format(sum(epoch_correct)/len(traindata)*100))


# lr = 0.00005
lr = 0.0001
window = 1024
optimizer_1 = optim.Adam(epilepsy_model.parameters(), lr=lr)
# optimizer_2 = optim.Adam(epilepsy_model_2.parameters(), lr=lr)

train(data_loader, 40, optimizer_1, epilepsy_model)
# train(data_loader, 5, optimizer_2, epilepsy_model_2)


def load_real_file(file_name):
    signals, signal_headers, header = pyedflib.highlevel.read_edf(filename)

