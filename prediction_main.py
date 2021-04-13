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
from sklearn.model_selection import train_test_split

from preprocess_music import get_dataset, spectrogram
from classifier_dataset import NoteDataset
from prediction_model import PredNet
from config import song_map, note_paths, note_map

# model params
lr = 0.005
batch_size = 12
max_epochs = 150
loss_fn = "BCEWithLogits"

# preprocess params
chunk_size_s = 1
pred_size_s = 0.8
overlap = 0
n_mels = 80
n_fft = 270

# misc
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)

song = "macdonald"
file_path = song_map[song]
net_state_path = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\wavenet_pytorch\model\classifier_bs10_lr0.002_epoch100.pt"
dataset, labels = get_dataset(file_path, net_state_path, chunk_size_s, overlap, n_mels=n_mels, n_fft=n_fft)
pred_size_s = round(chunk_size_s, 2)
data_train, data_valid, label_train, label_valid = train_test_split(dataset, labels, test_size=0.1, random_state=seed)


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

def get_complete_prediction(net, start_notes, sr, total_sec):
    # print(f"Getting {total_sec}s predicted output using {len(start_chunk)/sr}s of starting input...")
    # generate output `pred_size_s` given beginning chunk
    # spec_data = get_spec_data(start_chunk, sr, n_mels=n_mels, n_fft=n_fft)
    # output_preds = net(spec_data)
    # predicted_note = output_preds.argmax(axis=1)
    # predicted_notes = [note_map[predicted_note]]
    predicted_notes = start_notes

    # repeat for next chunks
    # prev_chunk = start_chunk.copy()
    predicted_note = note_map.index(start_notes[-1])
    for idx in trange(int(total_sec/pred_size_s)):
        waveform, sr = librosa.load(note_paths[note_map[predicted_note]])
        # mid_waveform = len(waveform)//2
        waveform = waveform[100:int(sr*pred_size_s) + 100]
        # waveform = waveform[mid_waveform:int(sr*pred_size_s)+mid_waveform] # get `pred_size_s` s of predicted waveform (use middle-ish)
        if idx == 0:
            prev_chunk = waveform.copy()
        curr_chunk = np.concatenate((prev_chunk, waveform))[-len(prev_chunk):] # get 1 sec of data total
        spec_data = get_spec_data(curr_chunk, sr, n_mels=n_mels, n_fft=n_fft)
        output_preds = net(spec_data)
        norm_out_preds = (output_preds[0] - min(output_preds[0]))/(max(output_preds[0])-min(output_preds[0])) # unity-based normalization to get output between 0,1
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
        waveform = waveform[20:int(sr * note_len)+20]
        # waveform = waveform[mid_waveform - int(sr * note_len)//2:int(sr * note_len)//2 + mid_waveform]
        output = np.concatenate((output, waveform)) if len(output) != 0 else waveform
    sf.write(outpath, output, sr)
    plt.figure()
    plt.plot(output)
    plt.savefig(outpath.split(".")[0] + ".png")
    return

def evaluate(net, loader, criterion):
    """ Evaluate the network on the validation set."""

    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0

    for i, data in enumerate(loader, 0):
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(input=outputs, target=labels)
        predictions = outputs.argmax(axis=1)
        err = predictions != labels.argmax(axis=1)
        total_err += int(err.sum())
        total_loss += loss.item()
        total_epoch += len(err)

    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)

    return err, loss

def main():
    train_err = np.zeros(max_epochs)
    train_loss = np.zeros(max_epochs)
    val_err = np.zeros(max_epochs)
    val_loss = np.zeros(max_epochs)

    net = PredNet()
    if loss_fn == "MSE":
        criterion = nn.MSELoss()
    elif loss_fn == "BCEWithLogits":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCELoss()
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

            loss = criterion(input=outputs.float(), target=labels.float())
            loss.backward()
            optimizer.step()

            predictions = outputs.argmax(axis=1)
            err = predictions != labels.argmax(axis=1)
            total_train_err += int(err.sum())
            total_train_loss += loss.item()
            total_epoch += len(err)

        train_err[epoch] = float(total_train_err) / (total_epoch)
        train_loss[epoch] = float(total_train_loss) / (idx + 1)

        val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)

        if epoch % 10 == 0:
            print(
                "Epoch {} | Train acc: {} | Train loss: {}".format(epoch + 1, 1 - train_err[epoch], train_loss[epoch]))

    # Plot
    plt.figure()
    plt.title("Training and Validation Accuracy over Epochs")
    plt.plot(np.arange(max_epochs), 1 - train_err, label="Training")
    plt.plot(np.arange(max_epochs), 1 - val_err, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("./images/prediction/TrainValAcc%s_bs%s_lr%s_epoch%s.png" % (song, batch_size, lr, max_epochs))

    plt.figure()
    plt.title("Training and Validation Loss over Epochs")
    plt.plot(np.arange(max_epochs), train_loss, label="Training")
    plt.plot(np.arange(max_epochs), val_loss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig("./images/prediction/TrainValLoss%s_bs%s_lr%s_epoch%s.png" % (song, batch_size, lr, max_epochs))

    # use final model and 1s start chunk to get full prediction
    waveform, sr = librosa.load(file_path)
    # start_offset_s = 0 #s offset from begining of source file
    # start_offset = int(start_offset_s * sr)
    # start_chunk = waveform[start_offset:int(sr*1)+start_offset]
    start_notes = ["G","G"] if song=="twinkle" else ["C"]
    note_sequence = get_complete_prediction(net, start_notes, sr, total_sec=15)
    print(f"Final note sequence (each chunk={pred_size_s}s): ", note_sequence)
    get_wav_output(f"./outputs/shortestchunk_{song}_ps{pred_size_s}s_epoch{max_epochs}_lr{lr}_bs{batch_size}_loss{loss_fn}.wav",
                   note_sequence, note_len=pred_size_s)

if __name__ == "__main__":
    main()