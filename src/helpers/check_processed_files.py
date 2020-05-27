from pathlib import Path
import mne
import os
from pyedflib import highlevel
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm


def import_files(folders=None, root=None):
    data = []
    if root is None:
        root = os.getcwd() + "/data/processed/"

    if folders is None:
        folders = ["normal/*.edf", "seizures/*.edf"]

    for i, folder in enumerate(folders):
        temp = ([(f, i) for f in sorted(Path(root).glob(folder))])
        data = data + temp

    return data


def detect_energy_spikes():
    seizure_file_1 = "/run/media/jmsvanrijn/3707BCE92020A60C/Data_2010_take_2/1.0.0/chb23/chb23_09.edf"
    normal_file = "/run/media/jmsvanrijn/3707BCE92020A60C/Data_2010_take_2/1.0.0/chb23/chb23_10.edf"
    start_time = 2000 # In seconds
    end_time = 3000 # In seconds
    hz = 256
    signals, signal_headers, header = highlevel.read_edf(str(seizure_file_1))

    recorded_seizures = np.array([0, 2589, 2660, 6885, 6947, 8505, 8532, 9580, 9664, len(signals[0])/hz])*hz
    seiz_23_1 = [29, 47]
    seiz_23_2 = [[30, 50], [53, 59]]
    seiz_23_3 = [2, 90]


    y_axis = np.resize([750, 0], len(recorded_seizures))

    z = np.array(signals[2])
    g = z[1:]**2 - z[1:]*z[:-1]
    g_2 = np.convolve(g, [1, 1, 1, 1, 1, 1, 1, 1])

    plt.subplot(211)
    plt.plot(np.transpose(z))
    plt.plot(recorded_seizures, y_axis, drawstyle="steps")

    y_axis = np.resize([np.max(g), 0], len(recorded_seizures))
    plt.subplot(212)
    plt.plot(np.transpose(g))
    plt.plot(np.transpose(g_2))
    plt.plot(recorded_seizures, y_axis, drawstyle="steps")
    plt.show()


def create_support_vector_machines():
    window_length = 1024
    all_files = import_files()

    # signals_1, signal_headers, header = highlevel.read_edf(str(all_files[0][0]))
    signals_2, signal_headers, header = highlevel.read_edf(str(all_files[180][0]))
    montage = mne.channels.make_standard_montage("standard_1020")

    # print(array(signals_1).shape)
    # signals_2 = np.array(signals_2)

    # clf = svm.SVC(kernel="linear")
    # clf.fit(signals_1[:, :window_length] + signals_2[:, :window_length], [0, 1])


detect_energy_spikes()