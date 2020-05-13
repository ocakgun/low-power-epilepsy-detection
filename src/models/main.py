# convolutional network model we will train to detect patterns in readings. For more information see my tutorial here.
# Found at; https://github.com/SamLynnEvans/EEG-grasp-and-lift

import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from src.models.epilepsy_data_loader import EpilepsyData
from src.models.cnn_model import convmodel
from src.models.learn_rate import find_lr
from src.models.train import train

import matplotlib.pyplot as plt

window_size = 1024
sample_spacing = 256
bs = 48
lr = 0.0005  # 67% lr = 0.00075 # 64%
EEG_DATA = "/home/jmsvanrijn/Documents/Afstuderen/Code/low-power-epilepsy-detection/data/processed/"

# Create a array seizures, 1= seizure, 0= normal
seizure_files_2 = [(str(seizure), str(1)) for seizure in sorted(Path(EEG_DATA).glob('seizures/*.edf'))]
normal_files_2 = [(str(normal), str(0)) for normal in sorted(Path(EEG_DATA).glob('normal/*.edf'))]
train_data = EpilepsyData(seizure_files_2 + normal_files_2, window_size)
train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)

epilepsy_model_1 = convmodel(window_size, 1).double()
optimizer_1 = optim.Adam(epilepsy_model_1.parameters(), lr=lr)
log, losses = find_lr(epilepsy_model_1, F.binary_cross_entropy, optimizer_1, train_loader, init_value=1e-8, final_value=10e-4, device="cpu")
plt.plot(log, losses)
plt.show()

# train(train_loader, 5, optimizer_1, epilepsy_model_1)

# optimizer_2 = optim.Adam(epilepsy_model_2.parameters(), lr=lr)
# epilepsy_model_2 = SingleLayerNN(23*window_size, 1).double()
# train(data_loader, 5, optimizer_2, epilepsy_model_2)
