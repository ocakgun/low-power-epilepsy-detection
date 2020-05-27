import numpy as np
import torch
from pyedflib import highlevel
from torch.utils.data import DataLoader
from src.models.complete_reading_loader import CompleteReading
from src.models.cnn_model import convmodel
from matplotlib import pyplot as plt

filename = "./models/model_1.pth"
bs = 64
hz = 256
sample_spacing = 256
window_size = 1024
rec_seiz = np.array([2589, 2660, 6885, 6947, 8505, 8532, 9580, 9664])

recorded = np.zeros(3691776//256)
recorded[rec_seiz[0]:rec_seiz[1]] = 1
recorded[rec_seiz[2]:rec_seiz[3]] = 1
recorded[rec_seiz[4]:rec_seiz[5]] = 1
recorded[rec_seiz[6]:rec_seiz[7]] = 1

model = convmodel(1024, 1)
model.load_state_dict(torch.load(filename))
model.double()
model.eval()

seizure_file_1 = "/run/media/jmsvanrijn/3707BCE92020A60C/Data_2010_take_2/1.0.0/chb23/chb23_09.edf"
load_data = CompleteReading(seizure_file_1, sample_spacing=256, window_size=1024)
load_loader = DataLoader(load_data, batch_size=bs)

time = 0  # seconds
out_data = []


for batch in load_loader:
    spacer = len(batch)
    time = time + spacer
    out = model(batch)
    round_out = torch.round(out)
    target = recorded[time-spacer: time]
    # total_correctnes = profile_results(round_out, target)
    out_data = np.concatenate((out_data, round_out.view(-1).detach().numpy().astype(int)))

x_axis = np.array(range(len(out_data)))*hz
signals, signal_headers, header = highlevel.read_edf(str(seizure_file_1))
rec_seizes = np.array([0, 2589, 2660, 6885, 6947, 8505, 8532, 9580, 9664, len(signals[0])/hz])*hz
y_axis = np.resize([750, 0], len(rec_seizes))

z = np.array(signals[2])
g = z[1:]**2 - z[1:]*z[:-1]

plt.subplot(211)
plt.plot(np.transpose(z))
# plt.plot(g)
plt.plot(x_axis, out_data*500, drawstyle="steps")
plt.plot(rec_seizes, y_axis, drawstyle="steps")

# plt.subplot(212)
# plt.plot(np.transpose(g))
# plt.plot(np.transpose(g_2))
# plt.plot(recorded_seizures, y_axis, drawstyle="steps")
plt.show()
