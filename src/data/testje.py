from pathlib import Path
import pyedflib
import numpy as np
import random

eeg_seizure_data = Path.cwd() / "data/interim/seizures"
seizure_path = "data/processed/seizures/"
normal_path = "data/processed/normal/"

files = sorted(Path(eeg_seizure_data).glob("*.edf"))

sig_len = []
sample_rate_ = []

for file in files:
    signals, signal_headers, header = pyedflib.highlevel.read_edf(str(file))
    sample_rate = signal_headers[0]["sample_rate"]
    length_sig = len(signals[0])
    loc = random.randint(0, length_sig - 1024)
    print("Length of signal:" + str(length_sig))
    print("Start location: " + str(loc))
    break

