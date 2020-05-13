from torch.utils.data import Dataset
from pyedflib import highlevel


class CompleteReading(Dataset):
    def __init__(self, file_name, sample_spacing, window_size):
        signals, signal_headers, header = highlevel.read_edf(file_name)

        self.sample_spacing = sample_spacing
        self.window_size = window_size
        self.signals = signals
        self.items = range(int(len(signals[0])/sample_spacing - window_size/sample_spacing - 1))
        self.length = len(self.items)

    def __getitem__(self, index):
        start = self.items[index]
        return self.signals[:, start*self.sample_spacing:(start+4)*self.sample_spacing]

    def __len__(self):
        return self.length
