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
import wget
import shutil
import random
import string
import mne


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # cut_seizures()
    cut_channels()


def cut_seizures():
    # Select the patients you want to check (available are; 1-24)
    PATIENTS = range(1, 24)

    # Make plots of patient
    PLOT = 0

    patient_folder = os.getenv("RAW_DATA_FOLDER")

    # Base url of data
    physioUrl = "https://physionet.org/files/chbmit/1.0.0/"

    # Download files and process the to raw to processed data, beware cost a lot of time
    DOWNLOAD_FILES = 0

    if DOWNLOAD_FILES:
        DOWNLOAD_FILES = yes_or_no_helper()

    if DOWNLOAD_FILES:
        download_in_cycles(physioUrl)
    else:
        seizure_timestamps, normal_recording_files = analyse_patients(patient_folder, PATIENTS)
        extract_seizure_edf(patient_folder, seizure_timestamps, normal_recording_files)


def yes_or_no_helper():
    yes = {"yes", "y", "ye", ""}
    no = {"no", "n"}

    print("Downloading files (50GB) and processing them takes a long time (+1 hour) are you sure? [y/n]")
    choice = input().lower()
    if choice in yes:
        return 1
    elif choice in no:
        return 0
    else:
        print("Please respond with 'yes' or 'no'")
        return yes_or_no_helper()


def download_in_cycles(baseUrl):
    raw_folder = "data/raw/"
    wget.download(baseUrl + "/RECORDS-WITH-SEIZURES", raw_folder)
    file = open(raw_folder + "RECORDS-WITH-SEIZURES", 'r')
    print(file.read())


def extract_seizure_edf(patient_folder, seizures, normal_files):
    t_general = time.time()
    t_delay = 60

    seizure_path = "data/interim/seizures/"
    normal_path = "data/interim/normal/"

    if os.path.isdir(seizure_path):
        shutil.rmtree(seizure_path, ignore_errors=True)
        shutil.rmtree(normal_path, ignore_errors=True)

    os.mkdir(seizure_path)
    os.mkdir(normal_path)

    for seizure in seizures:
        signals, signal_headers, header = pyedflib.highlevel.read_edf(patient_folder + seizure[0])
        signals = np.array(signals)

        sample_rate = signal_headers[0]["sample_rate"]
        start = seizure[1]*sample_rate
        end = seizure[2]*sample_rate
        seizure_signal = signals[:, start: end]
        normal_signal = signals[:, start+t_delay*sample_rate: end+t_delay*sample_rate]
        random_part = random.choice(string.ascii_letters) + random.choice(string.ascii_letters)

        seizure_filename = seizure_path + seizure[0].split("/")[1].split(".")[0] + "_" + random_part + "-seizure.edf"
        normal_filename = normal_path + seizure[0].split("/")[1].split(".")[0] + "_" + random_part + ".edf"

        # Some headers are broken, this fixes them
        for signal_header in signal_headers:
            if signal_header["digital_min"] == 0:
                signal_header["digital_min"] = -1
            if signal_header["digital_max"] == 0:
                signal_header["digital_max"] = 1

        pyedflib.highlevel.write_edf(seizure_filename, seizure_signal, signal_headers, header)
        pyedflib.highlevel.write_edf(normal_filename, normal_signal, signal_headers, header)

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


def load_edf_to_pd(edf_file):
    signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file)
    pd_signal_headers = []

    for signal_header in signal_headers:
        pd_signal_headers.append(signal_header["label"])

    pd_eeg = pd.DataFrame(data=signals, index=pd_signal_headers)
    return pd_eeg


def cut_channels():
    eeg_seizure_data = Path.cwd() / "data/interim/seizures"
    seizure_path = "data/processed/seizures/"
    normal_path = "data/processed/normal/"

    files = sorted(Path(eeg_seizure_data).glob("*.edf"))

    if os.path.isdir(seizure_path):
        shutil.rmtree(seizure_path, ignore_errors=True)
        shutil.rmtree(normal_path, ignore_errors=True)

    os.mkdir(seizure_path)
    os.mkdir(normal_path)

    headers = [(mne.io.read_raw_edf(f, preload=True).info["ch_names"]) for f in files]

    # We flag all channels which are bigger then expected
    for i, f in enumerate(headers):
        if len(f) > 23:
            drop_channels(files[i])
        else:
            load_right_channels(str(files[i]))


def drop_channels(file):
    amount_of_wanted_channels = 23
    unwanted_channels = ["--0", "--1", "--2", "--3", "--4", "--5", "-0", "-1", "-2", "-3", "-4", "-5", "ECG", "-", ".", "VNS", "LOC-ROC", "EKG1-CHIN", "C6-CS2", "C6"]
    wanted_channels = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8"]

    signals, signal_headers, header = pyedflib.highlevel.read_edf(str(file))
    signal_wanted_count = [(f["label"]) for f in signal_headers if f["label"] in wanted_channels]

    if len(signal_wanted_count) != amount_of_wanted_channels:
        signal_unwanted_count = [(f["label"]) for f in signal_headers if f["label"] in unwanted_channels]
        complete_list = [(f["label"]) for f in signal_headers]
        load_right_channels(edf_source=str(file), to_keep=list(set(complete_list) - set(signal_unwanted_count)))
    else:
        load_right_channels(edf_source=str(file), to_keep=signal_wanted_count)


def load_right_channels(edf_source, to_keep=None):
    load_seizure = edf_source
    save_seizure = edf_source.replace("interim", "processed")

    load_normal = edf_source.replace("-seizure", "").replace("seizures", "normal")
    save_normal = load_normal.replace("interim", "processed")

    # Seizures
    signals, signal_headers, header = pyedflib.highlevel.read_edf(load_seizure, ch_names=to_keep)
    if len(signal_headers) == 23:
        pyedflib.highlevel.write_edf(save_seizure, signals, signal_headers, header)

    # Normal
    signals, signal_headers, header = pyedflib.highlevel.read_edf(load_normal, ch_names=to_keep)
    if len(signal_headers) == 23:
        pyedflib.highlevel.write_edf(save_normal, signals, signal_headers, header)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
