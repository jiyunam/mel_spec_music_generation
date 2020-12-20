import numpy as np
from tqdm import trange, tqdm

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim

from NN_model import Net
from NN_dataset import Dataset
import time


def evaluate(net, loader, criterion):
    """ Evaluate the network on the validation set.
     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the average classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0

    for i, data in enumerate(loader, 0):
        inputs, labels = data
        labels = normalize_label(labels)  # Convert labels to 0/1
        outputs = net(inputs)
        loss = criterion(outputs, labels.float())
        corr = (outputs > 0.0).squeeze().long() != labels
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)

    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)

    return err, loss


def main():
    torch.manual_seed(1000)
    num_epochs = 10
    learning_rate = 0.01
    batch_size = 1
    dir_path = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\24_preludes_for_solo_piano"
    training_set = Dataset(dir_path)
    loadtr = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)

    ########################################################################
    device = 'cpu'
    if torch.cuda.is_available():
        print(f'Using {torch.cuda.get_device_name(0)}')
        device = 'cuda'

    # Define a Convolutional Neural Network, defined in model.py
    net = Net(device)

    if torch.cuda.is_available():
        net.cuda()

    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.5)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

    ########################################################################
    # Set up some numpy arrays to store the training/test loss/erruracy
    train_corr = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    # val_err = np.zeros(num_epochs)
    # val_loss = np.zeros(num_epochs)

    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_corr = 0.0

        total_epoch = 0
        idx = 0
        for inputs, target in tqdm(loadtr):
            target = target[:, :14000].to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = net(inputs.reshape(1,1,-1))
            outputs = outputs/outputs.mean() + 128
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # Calculate the statistics
            # corr = (outputs > 0.0).squeeze().long() != labels
            total_train_corr += (torch.isclose(outputs, target, atol=5)).sum()
            total_train_loss += loss.item()
            # print(loss.item(), outputs.max(), outputs.min())
            total_epoch += len(target[0])
            idx += 1

        train_corr[epoch] = float(total_train_corr) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (idx)
        # val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)

        print("Epoch {}: Train acc: {}, Train loss: {} ".format(epoch + 1, train_corr[epoch], train_loss[epoch]))

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    # Save the model to a file
    model_path = "model/1dconv"
    torch.save(net.state_dict(), model_path)


if __name__ == '__main__':
    main()
