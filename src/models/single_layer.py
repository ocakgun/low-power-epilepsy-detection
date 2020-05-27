import torch.nn as nn


class SingleLayerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # self.fc1 = nn.Linear(input_size, round(input_size/2), bias=True)
        # self.fc2 = nn.Linear(round(input_size/2), output_size, bias=True)
        self.fc1 = nn.Linear(input_size, output_size, bias=True)
        # self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        bs = x.size(0)
        out = x.view(bs, -1)
        out = self.fc1(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        out = self.sig(out)
        return out
