import json
import numpy as np
import mne
from scipy.stats import zscore

with open("artifacts/channels.json") as f:
    CHANNELS = json.load(f)

with open("artifacts/window_config.json") as f:
    cfg = json.load(f)

WINDOW_SIZE = cfg["window_size"]
STEP_SIZE = cfg["step_size"]

N_CHANNELS = len(CHANNELS)


def load_eeg(file_path):
    raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    raw.filter(1., 40.)

    data = raw.get_data()
    ch_names = raw.ch_names

    fixed = np.zeros((N_CHANNELS, data.shape[1]))

    for i, ch in enumerate(CHANNELS):
        if ch in ch_names:
            fixed[i] = data[ch_names.index(ch)]

    fixed = np.where(
        fixed.std(axis=1, keepdims=True) == 0,
        fixed,
        zscore(fixed, axis=1)
    )

    return fixed


def create_windows(data):
    windows = []
    for start in range(0, data.shape[1] - WINDOW_SIZE, STEP_SIZE):
        window = data[:, start:start + WINDOW_SIZE]
        windows.append(window.T)
    return np.array(windows, dtype=np.float32)
