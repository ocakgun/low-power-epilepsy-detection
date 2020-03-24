from typing import List
import numpy as np
import pyedflib
import pandas as pd
import os

ROOT = "/home/jmsvanrijn/Documents/Afstuderen/Code/30-01-2020_First_Try/"


def load_edf_to_pd(edf_file):
    signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file)
    pd_signal_headers = []

    for signal_header in signal_headers:
        pd_signal_headers.append(signal_header["label"])

    pd_eeg = pd.DataFrame(data=signals, index=pd_signal_headers)
    return pd_eeg


def pearson_correlation(pd_eeg):
    return pd_eeg.T.corr(method="pearson")


def list_files(patient_array):
    tot_seizure_recordings = []
    tot_normal_recordings = []

    for patient in patient_array:
        patient_folder = "chb0" + str(patient)
        recordings = os.listdir(ROOT + patient_folder)
        seizure_recordings: List[str] = []
        normal_recordings: List[str] = []

        for recording in recordings:
            if (".edf" and "seizures") in recording:
                seizure_recordings.append(ROOT + patient_folder + "/" + recording)
            elif ".edf" in recording:
                normal_recordings.append(ROOT + patient_folder + "/" + recording)

        tot_normal_recordings.append(normal_recordings)
        tot_seizure_recordings.append(seizure_recordings)

    return tot_seizure_recordings, tot_normal_recordings


def main():
    # Select the patients you want to check (available are; 1-24)
    PATIENTS = [1, 2]

    # Make plots of patient
    PLOT = 0

    seizures_recordings, normal_recordings = list_files(PATIENTS)
    pd_signal = load_edf_to_pd(normal_recordings[0][0])
    pearson_corr = pearson_correlation(pd_signal)

    # Reshapes to only contain the lowest values
    triu_pearson_corr = pearson_corr.mask(np.triu(np.ones(pearson_corr.shape, dtype=np.bool_)))
    print(triu_pearson_corr.min(0))


if __name__ == '__main__':
    main()
