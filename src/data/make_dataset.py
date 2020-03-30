# -*- coding: utf-8 -*-
import click
import logging
import os
import pyedflib
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    patient_folder = os.getenv("RAW_DATA_FOLDER")
    create_file_lists(patient_folder)


def create_file_lists(patient_folder):
    # Select the patients you want to check (available are; 1-24)
    PATIENTS = [1, 2]

    # Make plots of patient
    PLOT = 0

    seizures_recordings, normal_recordings = list_files(patient_folder, PATIENTS)

    print(seizures_recordings)

    # pd_signal = load_edf_to_pd(normal_recordings[0][0])
    # pearson_corr = pearson_correlation(pd_signal)
    #
    # # Reshapes to only contain the lowest values
    # triu_pearson_corr = pearson_corr.mask(np.triu(np.ones(pearson_corr.shape, dtype=np.bool_)))
    # print(triu_pearson_corr.min(0))


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
