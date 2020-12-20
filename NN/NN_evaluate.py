from NN_model import Net
from NN_dataset import Dataset
import torch

def load_model():
    net = Net()
    model_path = 'model/1dconv'
    net.load_state_dict(torch.load(model_path))
    return net

if __name__ == '__main__':
    net = load_model()
    dir_path = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\24_preludes_for_solo_piano"
    training_set = Dataset(dir_path)
    input, target = training_set[0]
    input = input.reshape(1,-1)
    target = target[:14000].reshape(1,-1)
    output = net(input)
