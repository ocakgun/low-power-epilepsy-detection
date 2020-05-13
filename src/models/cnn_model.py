import torch
import torch.nn as nn


class convmodel(nn.Module):
    def __init__(self, window_size, out_classes, drop=0.5, d_linear=124):
        super().__init__()

        self.conv2 = nn.Conv1d(23, 46, kernel_size=3, padding=0, stride=1)
        self.bn = nn.BatchNorm1d(46)
        self.pool = nn.MaxPool1d(2, stride=2)
        # self.linear1 = nn.Linear(5842, d_linear)

        self.linear1 = nn.Linear((round(window_size/2)-1)*46, d_linear)

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