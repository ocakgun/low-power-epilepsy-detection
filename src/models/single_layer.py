import torch.nn as nn


class SingleLayerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size, bias=False)
        self.relu = nn.ReLU()
        # self.softmin = nn.Softm(dim=-1)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        bs = x.size(0)
        out = x.view(bs, -1)
        out = self.fc1(out)
        out = self.relu(out)
        # out = self.softmax(out)
        # out = self.softmin(out)
        return out
