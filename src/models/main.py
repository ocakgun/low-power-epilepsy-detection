# convolutional network model we will train to detect patterns in readings. For more information see my tutorial here.
# Found at; https://github.com/SamLynnEvans/EEG-grasp-and-lift

import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from src.models.epilepsy_data_loader import EpilepsyData
from src.models.single_layer import SingleLayerNN
from src.models.cnn_model import convmodel
from src.models.learn_rate import find_lr
from src.models.train import train
import numpy as np
import random
from matplotlib import pyplot as plt

window_size = 1024
sample_spacing = 256
bs = 102
lr = 0.001 #0.0005  # 67% lr = 0.00075 # 64%
epochs = 10
train_ratio = 1

EEG_DATA = "/home/jmsvanrijn/Documents/Afstuderen/Code/low-power-epilepsy-detection/data/processed/"

# Create a array seizures, 1= seizure, 0= normal
seizure_files_2 = [(str(seizure), str(1)) for seizure in sorted(Path(EEG_DATA).glob('seizures/*.edf'))]
normal_files_2 = [(str(normal), str(0)) for normal in sorted(Path(EEG_DATA).glob('normal/*.edf'))]
all_files = seizure_files_2 + normal_files_2
all_files = random.sample(all_files, len(all_files))
train_data = EpilepsyData(all_files[:round(train_ratio*len(all_files))], window_size)
valid_data = EpilepsyData(all_files[round(train_ratio*len(all_files)):], window_size)
train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
# valid_data = DataLoader(valid_data, batch_size=bs, shuffle=True)

epilepsy_model_1 = convmodel(window_size, 1).double()
optimizer_1 = optim.Adam(epilepsy_model_1.parameters(), lr=lr)
# total_correctness, total_sensitivity, total_specificity, total_loss = train(train_loader, valid_data, epochs, optimizer_1, epilepsy_model_1, len(all_files), bs)
# log, losses = find_lr(epilepsy_model_1, F.binary_cross_entropy, optimizer_1, train_loader, init_value=1e-8, final_value=10e-4, device="cpu")
# total_correctness, total_sensitivity, total_specificity, total_loss = train(train_loader, 1, optimizer_1, epilepsy_model_1)

epilepsy_model_2 = SingleLayerNN(23*window_size, 1).double()
optimizer_2 = optim.Adam(epilepsy_model_2.parameters(), lr=lr)
total_correctness, total_sensitivity, total_specificity, total_loss = train(train_loader, valid_data, epochs, optimizer_2, epilepsy_model_2, len(all_files), bs)

length_data = len(train_data)
remainder = len(train_data) % bs
divide = (len(train_data)-remainder)/bs

tco = np.array(total_correctness)
tse = np.array(total_sensitivity)
tsp = np.array(total_specificity)
tlo = np.array(total_loss)

if not remainder:
    tco = sum(tco) * bs / length_data
else:
    tco = ((tco[:, :-1].sum(axis=1).reshape(epochs, 1)) * bs + tco[:, -1].reshape(epochs, 1) * remainder) / length_data
    tse = ((tse[:, :-1].sum(axis=1).reshape(epochs, 1)) * bs + tse[:, -1].reshape(epochs, 1) * remainder) / length_data
    tsp = ((tsp[:, :-1].sum(axis=1).reshape(epochs, 1)) * bs + tsp[:, -1].reshape(epochs, 1) * remainder) / length_data
    tlo = (tlo[:, :-1].sum(axis=1).reshape(epochs, 1)) + tlo[:, -1].reshape(epochs, 1)

# plt.plot(tlo, label="Correctness")
plt.plot(tco, label="Correctness")
plt.plot(tse, label="Sensitivity")
plt.plot(tsp, label="Specificity")

plt.show()
