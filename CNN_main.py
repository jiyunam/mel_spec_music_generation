import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import librosa
import numpy as np
import soundfile as sf

from tqdm import tqdm

from CNN_model import Net

sr = 22050
lr = 1e-5
epoch = 200
batch_size = 1
chunk_size_s = 3
overlap = 0
# chunk_size = int(chunk_size_s * sr)
#
# filename = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\24_preludes_for_solo_piano\jan_hanford-24_preludes_for_solo_piano-01-prelude_no__1_in_f_minor-30-59.mp3"
# waveform, sr_file = librosa.load(filename)
# num_chunks = int(1/overlap) * len(waveform) // chunk_size

def get_data(filename, chunk_size_s, overlap=0):
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

    # chunks = torch.FloatTensor(list(chunk_waveform(waveform.numpy().squeeze(), chunk_size, overlap)))
    chunks = torch.FloatTensor(list(chunk_waveform(waveform, chunk_size, overlap)))

    arr_specs = []
    for chunk in chunks:
        specgram = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=256)(chunk.reshape(1,-1))
        arr_specs.extend(np.array(specgram))

    return arr_specs

def get_batch(files):
    arr_specs = None
    idx = 0
    for file in tqdm(files):
        arr_spec = get_data(file, chunk_size_s, overlap)
        if idx == 0:
            arr_specs = np.empty(((len(files),) + np.array(arr_spec).shape))
        else:
            arr_specs[idx] = np.array(arr_spec)
        idx += 1

    train_specs = torch.FloatTensor(np.array(arr_specs))
    return train_specs

def train(epoch):
    dir_path = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\24_preludes_for_solo_piano"
    # dir_path = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\twinkle"
    files = glob.glob(os.path.join(dir_path, '*.mp3'))
    train_specs = get_batch(files)
    num_chunks = train_specs.shape[1]
    net = Net(num_chunks)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    # filename = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\24_preludes_for_solo_piano_wav\jan_hanford-24_preludes_for_solo_piano-01-prelude_no__1_in_f_minor-30-59.wav"
    # arr_specs = get_data(filename, chunk_size_s, overlap)
    # train_specs = torch.FloatTensor(np.array(arr_specs)).unsqueeze(1)

    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    for idx in range(epoch):
        # for t_idx, train_spec in enumerate(train_specs):
        t_idx = 0
        for train_spec in batch(train_specs, batch_size):
            if t_idx+1 >= len(train_specs):
                break
            optimizer.zero_grad()
            output = net(train_spec)
            # output = net(train_spec.unsqueeze(0))

            # inverse_mel_target = torchaudio.transforms.InverseMelScale(sample_rate=sr, n_stft=129, n_mels=80)(train_specs[t_idx+1])
            # target_audio = torchaudio.transforms.GriffinLim(n_fft=256)(inverse_mel_target)

            # output = output[0].detach()
            # output = output[:, :-1, :-1]
            # inverse_mel_pred = torchaudio.transforms.InverseMelScale(sample_rate=sr, n_stft=129, n_mels=80)(output)
            # pred_audio = torchaudio.transforms.GriffinLim(n_fft=256)(inverse_mel_pred)

            # loss = criterion(pred_audio.squeeze(), target_audio.squeeze())

            output = output[0][:, :-1, :-1]
            # loss = criterion(output, train_specs[t_idx + 1])
            loss = criterion(output, train_specs[t_idx + 1][0].unsqueeze(0))
            loss.backward()
            optimizer.step()
        print(f"Epoch:{idx}  Loss:{loss.item()}, Max output:{output.max().item()}")

    print("Saving last output")
    output = output.detach()
    inverse_mel_pred = torchaudio.transforms.InverseMelScale(sample_rate=sr, n_stft=129, n_mels=80)(output)
    pred_audio = torchaudio.transforms.GriffinLim(n_fft=256)(inverse_mel_pred)
    sf.write(f'output_CNN_s{chunk_size_s}_sr{sr}_bs{batch_size}_o{overlap}_lr{lr}_epoch{epoch}.wav',
             pred_audio.squeeze(), sr)
    return net



def main():
    net = train(epoch)
    print('done training')

    # filename = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\24_preludes_for_solo_piano_wav\jan_hanford-24_preludes_for_solo_piano-01-prelude_no__1_in_f_minor-30-59.wav"
    # arr_specs = get_data(filename, 2)
    # train_specs = torch.FloatTensor(np.array(arr_specs)).unsqueeze(1)

    # dir_path = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\24_preludes_for_solo_piano"
    # files = glob.glob(os.path.join(dir_path, '*.mp3'))
    # train_specs = get_batch(files)
    # output = net(train_specs[-1].unsqueeze(0))
    # output = output[0].detach()
    # output = output[:, :-1, :-1]
    # inverse_mel_pred = torchaudio.transforms.InverseMelScale(sample_rate=sr, n_stft=129, n_mels=80)(output)
    # pred_audio = torchaudio.transforms.GriffinLim(n_fft=256)(inverse_mel_pred)
    # sf.write(f'output_CNN_s{chunk_size_s}_sr{sr}_bs{batch_size}_o{overlap}_lr{lr}_epoch{epoch}.wav', pred_audio.squeeze(), sr)

if __name__ == '__main__':
    main()