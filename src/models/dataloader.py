import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from pathlib import Path

class EpilepsyData(Dataset):

    def __init__(self, path):
        super().__init__()
        # Get directory listing from path
        files = sorted(Path(EEG_DATA).glob('**/*.edf'))
        # Iterate through the listing and create a list of tuples (filename, label)
        self.items = [(str(f), f.name.split("-")[-1].replace(".edf", "")) for f in files]
        self.length = len(self.items)

    def __getitem__(self, index):
        filename, label = self.items[index]
        # audioTensor, rate = torchaudio.load(filename)
        audioTensor = ""
        return (audioTensor, int(label))

    def __len__(self):
        return self.length


bs = 12
EEG_DATA = Path.cwd() / "data/processed/"
train_epilepsy = EpilepsyData(EEG_DATA)
print(len(files))

loader = data.DataLoader(train_epilepsy, batch_size=bs, shuffle=True)