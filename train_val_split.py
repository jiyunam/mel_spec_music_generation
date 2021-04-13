import torch
import os
import numpy as np
import glob
import librosa
import torchaudio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

n_mels = 80
spec_len = 270
oneh_encoder = OneHotEncoder()

def get_data(dir_path):
    filenames = glob.glob(os.path.join(dir_path, '*.mp3'))
    dataset = torch.empty((len(filenames), n_mels, spec_len))
    labels = []
    for idx, filename in enumerate(filenames):
        data = torch.zeros(1, n_mels, spec_len)
        label = filename.split('\\')[-1].split('_')[0]
        waveform, sr = librosa.load(filename)
        specgram = torchaudio.transforms.MelSpectrogram(n_mels=n_mels, n_fft=spec_len)(torch.FloatTensor(waveform))
        data[:,:,:specgram.shape[-1]] = specgram #pad with zeros to size
        dataset[idx] = data
        labels.append(label)
    labels = np.asarray(labels)
    labels = oneh_encoder.fit_transform((labels).reshape(-1,1)).toarray()
    return dataset, labels

seed = 100
np.random.seed(seed)
torch.manual_seed(seed)

dir_path = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\notes"
dataset, labels = get_data(dir_path)

data_train, data_valid, label_train, label_valid = train_test_split(dataset, labels, test_size=0.1, random_state=seed)
# for now use entire dataset for train
# data_train, data_valid, label_train, label_valid = dataset, None, labels, None
