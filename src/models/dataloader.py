from pathlib import Path

#
# class EpilepsyData(Dataset):
#     def __init__(self):
#         # Get directory listing from path
#         files = Path(path).glob("*.wav")
#         print(files)
#         # Iterate through the listing and create a list of tuples (filename, label)
        self.items = [(str(f), f.name.split("-")[-1].replace(".wav", "")) for f in files]
#         self.length = len(self.items)
#
#     def __getitem__(self, index):
#         filename, label = self.items[index]
#         # Find normal
#         # audioTensor, rate = torchaudio.load(filename)
#         return (audioTensor, int(label))
#
#     def __len__(self):
#         return self.length

bs = 12
EEG_DATA = Path.cwd() / "data/processed/"

