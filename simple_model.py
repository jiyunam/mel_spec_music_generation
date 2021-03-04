import soundfile as sf
import scipy.stats as stats
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data

from dataset import Dataset

dir_path = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\24_preludes_for_solo_piano"
seed = 42 # or np.random.seed
batch_size = 1
# window_size = 100000
predict_percentage = 0.1

training_set = Dataset(dir_path)
loadtr = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=seed)
mse_loss = nn.MSELoss()

for filename, ytrain, sr in loadtr:
    bit_rate = sr.numpy()[0]
    predict_size = int(len(ytrain[0]) * predict_percentage)
    # x = np.arange(0, predict_size)
    x = np.linspace(0, 2 * np.pi, predict_size)

    input_ytrain = ytrain[0][:-predict_size]
    output_ytrain = ytrain[0][-predict_size:]

    # Predict just using mean value
    predict_val = torch.Tensor.float(input_ytrain).mean()
    predict_ytrain = (torch.tensor(np.sin(x)) * predict_val) + 128
    decoded = training_set.mu_law_decode(predict_ytrain.numpy().astype('int'))
    loss = mse_loss(predict_ytrain, output_ytrain)
    print(loss)
    sf.write('mean_sine.wav', decoded, sr.numpy()[0])

    # Predict using normal distribution estimate
    lower, upper = 0, 255 # set lower and upper limits
    mu, sigma = stats.norm.fit(input_ytrain)
    N = 10 # number of notes to pick out
    predict_ytrain = stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=N)
    # repeat values to have note maintenance
    predict_ytrain = np.repeat(predict_ytrain, np.ceil(predict_size/N))
    predict_ytrain = predict_ytrain[:predict_size] # make sure size is correct
    predict_ytrain = torch.tensor(predict_ytrain)
    predict_ytrain = (torch.tensor(np.sin(x)) * predict_ytrain) + 128
    decoded = training_set.mu_law_decode(predict_ytrain.numpy().astype('int'))
    loss = mse_loss(predict_ytrain, output_ytrain)
    print(loss)
    sf.write('normal_dist_sampled_sine.wav', decoded, sr.numpy()[0])
    print('yes')

    N = 50
    frequency = 500 #Hz, waves per second, 261.63=C4-note.
    x = np.arange(0, predict_size)
    predict_ytrain = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=N)
    predict_ytrain = np.repeat(predict_ytrain, np.ceil(predict_size / N))
    predict_ytrain = predict_ytrain[:predict_size]  # make sure size is correct
    predict_ytrain = np.sin(x / ((bit_rate / frequency) / np.pi))*predict_ytrain
    decoded = training_set.mu_law_decode(predict_ytrain.astype('int'))
    sf.write('normal_dist_sampled_freq.wav', decoded, bit_rate)


    # # break up training tensor into chunks of window
    # split_ytrain = torch.split(ytrain[0], window_size)
    #
    # # predict `predict_percentage` of window based on window-window*predict_percentage
    # avg_loss, avg_acc = 0, 0
    # for chunk in split_ytrain:
    #     input_ytrain = chunk[:-predict_size]
    #     output_ytrain = chunk[-predict_size:].type(torch.FloatTensor)
    #
    #     # get mean of `input_ytrain` and output that as prediction
    #     predict_val = torch.Tensor.float(input_ytrain).mean()
    #     predict_ytrain = torch.ones(output_ytrain.shape) * predict_val
    #     predict_ytrain = Variable(predict_ytrain, requires_grad=True).type(torch.FloatTensor)
    #
    #     # compare output with prediction (MSE loss)
    #     avg_acc += float(float(torch.sum(predict_ytrain == output_ytrain)) / float(predict_ytrain.shape[0]))
    #     loss = mse_loss(predict_ytrain, output_ytrain)
    #     loss.backward()
    #     avg_loss += float(loss)
    # print(f"Avg Acc: {avg_acc/len(split_ytrain)}   Avg Loss:{avg_loss/len(split_ytrain)}")
