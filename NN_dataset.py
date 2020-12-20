import librosa as lr
import glob, os
import torch
import numpy as np
from torch.utils import data
import torchaudio

class Dataset(data.Dataset):
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.files = glob.glob(os.path.join(file_dir, '*.mp3')) # currently just mp3

    def __len__(self):
        # length of dataset
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        y, sr = lr.load(file)
        # S = lr.feature.melspectrogram(y, sr=sr, n_mels=128)
        # log_S = lr.power_to_db(S, ref=np.max)
        y = torch.from_numpy(y)  # convert to tensor
        # specgram = torchaudio.transforms.Spectrogram()(y)
        y = torchaudio.transforms.MuLawEncoding()(y)
        # split y
        # freq_len, times_len = S.shape
        # times_len = len(y)
        # subsample
        y = y.unfold(0,1,5).squeeze().float()
        split_val = 114000
        input = y[:split_val]
        target = y[split_val:]
        return input, target
