# -*- coding: utf-8 -*-
import logging
import os
import pyedflib
import pandas as pd
import re
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import time


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    patient_folder = os.getenv("RAW_DATA_FOLDER")
    create_file_lists(patient_folder)


def create_file_lists(patient_folder):
    # Select the patients you want to check (available are; 1-24)
    PATIENTS = range(1, 23)

    # Make plots of patient
    PLOT = 0

    # seizures_recordings, normal_recordings = list_files(patient_folder, PATIENTS)
    seizure_timestamps, normal_recording_files = analyse_patients(patient_folder, PATIENTS)

    extract_seizure_edf(patient_folder, seizure_timestamps, normal_recording_files)

    # pearson_corr = pearson_correlation(pd_signal)
    #
    # # Reshapes to only contain the lowest values
    # triu_pearson_corr = pearson_corr.mask(np.triu(np.ones(pearson_corr.shape, dtype=np.bool_)))
    # print(triu_pearson_corr.min(0))


def extract_seizure_edf(patient_folder, seizures, normal_files):
    t_general = time.time()
    print(seizures)
    for seizure in seizures:
        signals, signal_headers, header = pyedflib.highlevel.read_edf(patient_folder + seizure[0])
        signals = np.array(signals)

        sample_rate = signal_headers[0]["sample_rate"]
        start = seizure[1]*sample_rate
        end = seizure[2]*sample_rate
        seizure_signal = signals[:, start: end]
        normal_signal = signals[:, start+60*sample_rate: end+60*sample_rate]

        print(seizure)
        filename = "extract-seizures/" + seizure[0].split("/")[1].split(".")[0] + "-seizure.edf"
        print(filename)

        pyedflib.highlevel.write_edf(filename, seizure_signal, signal_headers, header)
        # normal_edf = pyedflib.highlevel.write_edf("edf_normal_file", normal_signal, signal_headers, header)


        # mne.set_log_level("WARNING")
        # raw = mne.io.read_raw_edf(patient_folder + seizure[0], preload=True)
        # print(raw.info["ch_names"])

        # raw.set_montage("standard_1020", match_case=False, verbose=True)

        # raw.plot()
        # plt.show()


        # montage = mne.channels.make_standard_montage("standard_1020")
        print("Check")

    elapsed = time.time() - t_general
    print(elapsed)
def analyse_patients(ROOT, patient_array):
    seizure_data = []
    normal_data = []

    for patient in patient_array:
        normal_data_patient = []

        if patient < 10:
            patient_folder = "chb0" + str(patient)
        else:
            patient_folder = "chb" + str(patient)

        summary = open(ROOT + patient_folder + "/" + patient_folder + "-summary.txt", "r")
        summary = summary.readlines()

        sampling_rate = int(re.findall(r'\d+', summary[0].strip())[0])

        for num, line in enumerate(summary):
            normal_data_patient_per_file = []
            if not(line.find("File Name: ")):
                file_number = summary[num].strip()
                start_time = summary[num+1].strip()[-8:]
                time_recording = clock_to_timestamp(summary[num+2].strip()) - clock_to_timestamp(summary[num+1].strip())
                num_of_seizures = int(re.findall(r'\d+', summary[num + 3].strip())[0])

                normal_data_patient_per_file.append(patient_folder + "/" + file_number.split()[-1])
                normal_data_patient_per_file.append(start_time)
                normal_data_patient_per_file.append(time_recording)
                normal_data_patient_per_file.append(sampling_rate)
                normal_data_patient.append(normal_data_patient_per_file)

                # Select line with start and end of seizure, strip spaces, regex on numbers and cast to int
                for seizures in range(num_of_seizures):
                    seizure_data_patient = []
                    seizure_data_patient.append(patient_folder + "/" + file_number.split()[-1])
                    seizure_data_patient.append(int(re.findall(r'\d{2,}', summary[num+4+2*seizures].strip())[0]))
                    seizure_data_patient.append(int(re.findall(r'\d{2,}', summary[num+5+2*seizures].strip())[0]))
                    seizure_data.append(seizure_data_patient)
        normal_data.append(normal_data_patient)
    return seizure_data, normal_data


def clock_to_timestamp(clock_time):
    seconds = int(clock_time[-2:])
    minutes = int(clock_time[-5:-3])
    hours = int(clock_time[-8:-6])

    return seconds + minutes*60 + hours*60*60


def edf_to_wav(edf_file):
    signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file)


def load_edf_to_pd(edf_file):
    signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file)
    pd_signal_headers = []

    for signal_header in signal_headers:
        pd_signal_headers.append(signal_header["label"])

    pd_eeg = pd.DataFrame(data=signals, index=pd_signal_headers)
    return pd_eeg


def pearson_correlation(pd_eeg):
    return pd_eeg.T.corr(method="pearson")


def list_files(ROOT, patient_array):
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


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
