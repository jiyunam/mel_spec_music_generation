import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import soundfile as sf

from tqdm import tqdm

from CNN.CNN_model import Net
from CNN.CNN_data import get_data

sr = 22050
lr = 1e-6
epoch = 250
batch_size = 1
chunk_size_s = 3
overlap = 0
lowcut = 20
highcut = 11000
# chunk_size = int(chunk_size_s * sr)

album = 'twinkle'
if album=="piano":
    dir_path = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\24_preludes_for_solo_piano"
elif album=="twinkle":
    dir_path = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\twinkle"
# filename = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\24_preludes_for_solo_piano\jan_hanford-24_preludes_for_solo_piano-01-prelude_no__1_in_f_minor-30-59.mp3"
# waveform, sr_file = librosa.load(filename)
# num_chunks = int(1/overlap) * len(waveform) // chunk_size

def get_batch(files):
    arr_specs = None
    idx = 0
    for file in tqdm(files):
        arr_spec = get_data(file, chunk_size_s, overlap, lowcut, highcut)
        if idx == 0:
            arr_specs = np.empty(((len(files),) + np.array(arr_spec).shape))
        else:
            arr_specs[idx] = np.array(arr_spec)
        idx += 1

    train_specs = torch.FloatTensor(np.array(arr_specs))
    return train_specs

def train(epoch):
    files = glob.glob(os.path.join(dir_path, '*.mp3'))
    train_specs = get_batch(files)
    net = Net()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # filename = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\24_preludes_for_solo_piano_wav\jan_hanford-24_preludes_for_solo_piano-01-prelude_no__1_in_f_minor-30-59.wav"
    # arr_specs = get_data(filename, chunk_size_s, overlap)
    # train_specs = torch.FloatTensor(np.array(arr_specs)).unsqueeze(1)

    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    for idx in range(epoch):
        # for t_idx, train_spec in enumerate(train_specs):
        # t_idx = batch_size
        for train_spec in batch(train_specs, batch_size):
            # if t_idx+batch_size >= len(train_specs):
            #     break
            for time_chunk_idx in range(train_spec.shape[1]-1):
                optimizer.zero_grad()
                output = net(train_spec[:,time_chunk_idx, :,:].unsqueeze(0))

                output = output[0][:, :-1, :-1]
                loss = criterion(output, train_spec[:,time_chunk_idx+1, :,:])
                # loss = criterion(output, train_specs[t_idx + 1][0].unsqueeze(0))
                loss.backward()
                optimizer.step()
                # t_idx += batch_size
        print(f"Epoch:{idx}  Loss:{loss.item()}, Max output:{output.max().item()}")

    print("Saving last output")
    output = output.detach()
    inverse_mel_pred = torchaudio.transforms.InverseMelScale(sample_rate=sr, n_stft=129, n_mels=80)(output)
    pred_audio = torchaudio.transforms.GriffinLim(n_fft=256)(inverse_mel_pred)
    sf.write(f'{album}_CNN_s{chunk_size_s}_sr{sr}_bs{batch_size}_o{overlap}_'
             f'bp[{lowcut},{highcut}]_lr{lr}_epoch{epoch}.wav',
             pred_audio[-1], sr)
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