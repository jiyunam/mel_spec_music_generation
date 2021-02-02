import torch
import librosa
import torchaudio
import numpy as np
from scipy.signal import butter, lfilter

def get_data(filename, chunk_size_s, overlap=0, lowcut=None, highcut=None):
    # waveform, sr = torchaudio.load(filename)
    waveform, sr = librosa.load(filename)

    # split waveform into chunks
    chunk_size = int(chunk_size_s * sr)

    def chunk_waveform(waveform, chunk_size, overlap):
        idx = 0
        # overlap_chunk = 0
        overlap_chunk = int(overlap * chunk_size)
        while idx + chunk_size - overlap_chunk <= len(waveform):
            yield waveform[idx:idx + chunk_size - overlap_chunk]
            idx += chunk_size - overlap_chunk

    if lowcut and highcut:
        waveform = butter_bandpass_filter(waveform, lowcut, highcut, sr)

    chunks = torch.FloatTensor(list(chunk_waveform(waveform, chunk_size, overlap)))

    arr_specs = []
    for chunk in chunks:
        specgram = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=256)(chunk.reshape(1,-1))
        arr_specs.extend(np.array(specgram))

    return arr_specs

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

