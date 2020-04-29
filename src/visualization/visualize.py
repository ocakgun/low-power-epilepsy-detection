import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import matplotlib as plt
import mne
import pickle

def main():
    logger = logging.getLogger(__name__)

    # pearson_corr = pearson_correlation(pd_signal)
    #
    # # Reshapes to only contain the lowest values
    # triu_pearson_corr = pearson_corr.mask(np.triu(np.ones(pearson_corr.shape, dtype=np.bool_)))
    # print(triu_pearson_corr.min(0))


    # count_channels()
    small_test()


def small_test():
    with open("/home/jmsvanrijn/Documents/Afstuderen/pkl/seiz_0_1.pkl", "rb") as f:
        date = pickle.load(f)
        print(len(date.data[0][0]))
        print(len(date.data[0]))
        print(len(date.data))

    with open("/home/jmsvanrijn/Documents/Afstuderen/pkl/seiz_0_2.pkl", "rb") as g:
        date = pickle.load(g)
        print(len(date.data[0]))
        print(len(date.data))

def count_channels():
    eeg_seizure_data = Path.cwd() / "data/processed/seizures"
    files = sorted(Path(eeg_seizure_data).glob("*.edf"))

    headers = [(mne.io.read_raw_edf(f, preload=True).info["ch_names"]) for f in files]

    [print(len(header)) for header in headers if len(header) != 23]


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


