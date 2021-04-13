# Main module for note classification task
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classifier_dataset import NoteDataset
from classifier_model import Net
from train_val_split import data_train, data_valid, label_train, label_valid

lr = 0.002
batch_size = 10
max_epochs = 100

def load_data(batch_size):
    train_dataset = NoteDataset(data_train, label_train)
    valid_dataset = NoteDataset(data_valid, label_valid)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

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

    net = Net()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train_loader, val_loader = load_data(batch_size)
    pred_pd, target_pd = pd.DataFrame([]), pd.DataFrame([])
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

        val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)

        pred_row = pd.DataFrame([predictions.numpy()], index=[epoch])
        target_row = pd.DataFrame([labels.argmax(axis=1).numpy()], index=[epoch])
        pred_pd = pd.concat([pred_pd, pred_row]) if not pred_pd.empty else pred_row
        target_pd = pd.concat([target_pd, target_row]) if not target_pd.empty else target_row

        if epoch%10 == 0:
            print("Epoch {} | Train acc: {} | Train loss: {}".format(epoch + 1, 1 - train_err[epoch], train_loss[epoch]))
            # print("outputs: ", outputs)
            # print("predictions: ", predictions)
            # print("target:      ", labels.argmax(axis=1))

    torch.save(net.state_dict(), f"model/classifier_bs{batch_size}_lr{lr}_epoch{max_epochs}.pt")
    # pred_pd.to_csv("predicted_notes.csv")
    # target_pd.to_csv("target_notes.csv")
    plt.figure()
    plt.title("Training and Validation Accuracy over Epochs")
    plt.plot(np.arange(max_epochs), 1 - train_err, label="Training")
    plt.plot(np.arange(max_epochs), 1 - val_err, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("./images/classification/TrainValAcc_bs%s_lr%s_epoch%s.png" %(batch_size, lr, max_epochs))

    plt.figure()
    plt.title("Training and Validation Loss over Epochs")
    plt.plot(np.arange(max_epochs), train_loss, label="Training")
    plt.plot(np.arange(max_epochs), val_loss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig("./images/classification/TrainValLoss_bs%s_lr%s_epoch%s.png" %(batch_size, lr, max_epochs))

if __name__ == "__main__":
    main()
