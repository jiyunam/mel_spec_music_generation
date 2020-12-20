import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from tqdm import tqdm

from model import WaveNet
from data_processing import Dataset

# Model Parameters
layers = 10
blocks = 1
classes = 256
residual_dim = 128
skip_dim = 512
kernel_size = 1

# Training Parameters
dir_path = r"C:\Users\jiyun\Desktop\Jiyu\2020-2021\ESC499 - Thesis\WaveNet\magnatagatune\data\24_preludes_for_solo_piano"
max_epochs = 10
batch_size = 1
learning_rate = 1e-3
seed = 42 # or np.random.seed

# Get Model
device = 'cpu'
if torch.cuda.is_available():
    print(f'Using {torch.cuda.get_device_name(0)}')
    device = 'cuda'
    
model = WaveNet(layers=layers, blocks=blocks, classes=classes, 
                residual_dim=residual_dim, skip_dim=skip_dim, 
                kernel_size=kernel_size, device=device)

if torch.cuda.is_available():
    model.cuda()
    
sample_size = 16000
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


# Get Training Set
training_set = Dataset(dir_path)
loadtr = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=seed)

# Training Loop
def train(epoch):
    for filename, ytrain in loadtr:
        startx = 0
        idx = np.arange(startx + model.rf, ytrain.shape[-1] - model.rf- sample_size, sample_size)
        
        count, avg_loss,avg_acc = 0, 0, 0
        model.train()
        
        for i, ind in enumerate(idx):
            optimizer.zero_grad()
            target0 = ytrain[:, ind - model.rf:ind + sample_size - 1].to(device)
            target1 = ytrain[:, ind:ind + sample_size].to(device)
            
            output = model(target0)
            a = output.max(dim=1, keepdim=True)[1].view(-1)
            b = target1.view(-1)
            assert (a.shape[0] == b.shape[0])
            
            avg_acc += float(float(torch.sum(a.long() == b.long())) / float(a.shape[0]))
            loss = criterion(output, target1)
            loss.backward()
            optimizer.step()
            avg_loss += float(loss)
            
#             if(float(loss) > 10):print(float(loss))
            count += 1
            
        print('Epoch{}: loss for train:{:.4f}, acc:{:.4f}'.format(epoch, avg_loss/count, avg_acc/count))
        
def test(epoch):  # testing data
    model.eval()
    with torch.no_grad():
        for filename, y, queue in loadval:
            music = model.slowInfer(queue, device, y, 16000 * 3)
            print(music[:1000])
            ans0 = mu_law_decode(music.numpy().astype('int'))

            if not os.path.exists('vsCorpus/'): os.makedirs('vsCorpus/')
            write(savemusic.format(epoch), sample_rate, ans0)
            print('test stored done', np.round(time.time() - start_time))        
        
if __name__ == '__main__':
    for epoch in tqdm(range(max_epochs)):
        train(epoch)
#         if epoch % 4 == 0 and epoch > 0: 
#             test(epoch)