# Main module for note prediction task
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm

from preprocess_music import get_dataset, spectrogram
from classifier_dataset import NoteDataset
from prediction_model import PredNet

# model params
lr = 0.01
batch_size = 12
max_epochs = 600

# preprocess params
chunk_size_s = 1
pred_size_s = 1
overlap = 0
n_mels = 80
n_fft = 270

# misc
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)

file_path = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\twinkle\twinkle-piano.mp3"
net_state_path = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\wavenet_pytorch\model\classifier_bs10_lr0.002_epoch100.pt"
data_train, label_train = get_dataset(file_path, net_state_path, chunk_size_s, overlap, n_mels=n_mels, n_fft=n_fft)
data_valid, label_valid = None, None

note_paths = {
    "A": r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\twinkle_notes\A_1.mp3",
    "C": r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\twinkle_notes\C_3.mp3",
    "D": r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\twinkle_notes\D_1.mp3",
    "E": r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\twinkle_notes\E_1.mp3",
    "F": r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\twinkle_notes\F_1.mp3",
    "G": r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\twinkle_notes\G_1.mp3"
}
note_map = ["A","C","D","E","F","G"]

def load_data(batch_size):
    train_dataset = NoteDataset(data_train, label_train)
    valid_dataset = NoteDataset(data_valid, label_valid)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_spec_data(waveform, sr, *, n_mels=n_mels, n_fft=n_fft):
    spec = spectrogram(waveform, sr, chunk_size_s=None, overlap=0, n_mels=n_mels, n_fft=n_fft)
    spec_data = torch.zeros(spec.shape[0], n_mels, n_fft)
    spec_data[:, :, :spec.shape[-1]] = spec  # PADDING WITH ZEROS
    return spec_data

def get_complete_prediction(net, start_chunk, sr, total_sec):
    print(f"Getting {total_sec}s predicted output using {len(start_chunk)/sr}s of starting input...")
    # generate output `pred_size_s` given beginning chunk
    spec_data = get_spec_data(start_chunk, sr, n_mels=n_mels, n_fft=n_fft)
    output_preds = net(spec_data)
    predicted_note = output_preds.argmax(axis=1)
    predicted_notes = [note_map[predicted_note]]

    # repeat for next chunks
    prev_chunk = start_chunk.copy()
    for _ in trange(int(total_sec/pred_size_s)):
        waveform, sr = librosa.load(note_paths[note_map[predicted_note]])
        mid_waveform = len(waveform)//2
        waveform = waveform[mid_waveform:int(sr*pred_size_s)+mid_waveform] # get `pred_size_s` s of predicted waveform (use middle-ish)
        curr_chunk = np.concatenate((prev_chunk, waveform))[-len(prev_chunk):] # get 1 sec of data total
        spec_data = get_spec_data(curr_chunk, sr, n_mels=n_mels, n_fft=n_fft)
        output_preds = net(spec_data)
        norm_out_preds = (output_preds[0] - min(output_preds[0]))/(max(output_preds[0])-min(output_preds[0])) #unity-based normalization to get output between 0,1
        norm_out_preds /= norm_out_preds.sum() # sum to 1 for prob distribution
        # predicted_note = output_preds.argmax(axis=1)
        predicted_note = np.random.choice(len(norm_out_preds), 1, p=norm_out_preds.detach().numpy())[0]
        predicted_notes.append(note_map[predicted_note])
        prev_chunk = curr_chunk.copy()
    return predicted_notes

def get_wav_output(outpath, predicted_notes, note_len=0.25):
    print(f"Writing output to {outpath}...")
    output = np.array([])
    for predicted_note in tqdm(predicted_notes):
        waveform, sr = librosa.load(note_paths[predicted_note])
        mid_waveform = len(waveform) // 2
        waveform = waveform[mid_waveform:int(sr * note_len) + mid_waveform]
        output = np.concatenate((output, waveform)) if len(output) !=0 else waveform
    sf.write(outpath, output, sr)
    plt.figure()
    plt.plot(output)
    plt.savefig(outpath.split(".")[0] + ".png")
    return

def main():
    train_err = np.zeros(max_epochs)
    train_loss = np.zeros(max_epochs)

    net = PredNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_loader, val_loader = load_data(batch_size)
    for epoch in range(max_epochs):
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        for idx, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = net(inputs)

            loss = criterion(input=outputs, target=labels)
            loss.backward()
            optimizer.step()

            predictions = outputs.argmax(axis=1)
            err = predictions != labels.argmax(axis=1)
            total_train_err += int(err.sum())
            total_train_loss += loss.item()
            total_epoch += len(err)

        train_err[epoch] = float(total_train_err) / (total_epoch)
        train_loss[epoch] = float(total_train_loss) / (idx + 1)

        if epoch % 10 == 0:
            print(
                "Epoch {} | Train acc: {} | Train loss: {}".format(epoch + 1, 1 - train_err[epoch], train_loss[epoch]))

    # use final model and 1s start chunk to get full prediction
    waveform, sr = librosa.load(file_path)
    start_offset_s = 0 #s offset from begining of source file
    start_offset = int(start_offset_s * sr)
    start_chunk = waveform[start_offset:int(sr*1)+start_offset]
    note_sequence = get_complete_prediction(net, start_chunk, sr, total_sec=20)
    print(f"Final note sequence (each chunk={pred_size_s}s): ", note_sequence)
    get_wav_output(f"./outputs/ps{pred_size_s}s_epoch{max_epochs}_lr{lr}_bs{batch_size}.wav",
                   note_sequence, note_len=pred_size_s)

if __name__ == "__main__":
    main()