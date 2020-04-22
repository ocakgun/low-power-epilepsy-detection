import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import matplotlib as plt
import mne
import os
import numpy as np
import pandas as pd
from collections import Counter

def main():
    logger = logging.getLogger(__name__)

    # pearson_corr = pearson_correlation(pd_signal)
    #
    # # Reshapes to only contain the lowest values
    # triu_pearson_corr = pearson_corr.mask(np.triu(np.ones(pearson_corr.shape, dtype=np.bool_)))
    # print(triu_pearson_corr.min(0))
    count_channels()


def count_channels():
    eeg_seizure_data = Path.cwd() / "data/processed/seizures"
    files = sorted(Path(eeg_seizure_data).glob("*.edf"))

    headers = [(mne.io.read_raw_edf(f, preload=True).info["ch_names"]) for f in files]
    headers_flatten = np.concatenate(np.array(headers)).ravel()

    # We flag all channels which are bigger then expected
    bigger_then_23_channels = [[i, len(f)] for i, f in enumerate(headers) if len(f) > 23]
    print(bigger_then_23_channels)
    print("Amount of files: " + str(len(files)))
    print(str(len(files)) + " files contain a total of " + str(len(headers_flatten)) + " channels. Which is: " + str(len(headers_flatten)/len(files)))
    # print(Counter(headers_flatten))

    for i in bigger_then_23_channels:
        raw = mne.io.read_raw_edf(files[i[0]], preload=True).info["ch_names"]
        print(files[i[0]])
        print(len(raw))
        print(raw)
        break


def plot_eeg():
    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_edf(patient_folder + seizure[0], preload=True)
    print(raw.info["ch_names"])

    raw.set_montage("standard_1020", match_case=False, verbose=True)

    raw.plot()
    plt.show()

    montage = mne.channels.make_standard_montage("standard_1020")

def pearson_correlation(pd_eeg):
    return pd_eeg.T.corr(method="pearson")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


