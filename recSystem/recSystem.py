import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import torch
import requests
import shutil, os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

train_on_gpu = torch.cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

# check if ./model exists
if os.path.exists('/data/model'):
    shutil.rmtree('/data/model')
os.makedirs('/data/model')

# get the dataset url from the environment variable
dataset_url = os.environ.get('DATASET_URL', 'https://homepages.dcc.ufmg.br/~cunha/hosted/cloudcomp-2023s2-datasets/2023_spotify_ds1.csv')
retry = 0
while True:
    response = requests.get(dataset_url, verify=False)
    if response.status_code == 200:
        with open('/data/2023_spotify_ds1.csv', 'wb') as f:
            f.write(response.content)
            break

    else:
        print('Failed to get the dataset, retrying...{retry}')
        retry += 1
        if retry > 5:
            print('Failed to get the dataset after 5 retries')
            sys.exit(1)
        
playlists = pd.read_csv('/data/2023_spotify_ds1.csv')
playlists = playlists[['pid', 'track_name']]

# # get the new playlist from user
# playlist2 = pd.read_csv('./Dataset/2023_spotify_ds2.csv')
# playlist2 = playlist2[['pid', 'track_name']]
# lucky_pid = playlist2.pid.sample(1).values[0]
# new_playlist = playlist2[playlist2.pid == lucky_pid]

# # update the playlist
# playlists = pd.concat([playlists, new_playlist])

num_epochs = 10
learning_rate = 0.01
batch_size = 128
n_users = len(playlists.pid.unique())
n_items = len(playlists.track_name.unique())
n_factors = 20

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=50):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.user_factors.weight.data.uniform_(0,0.05)
        self.item_factors.weight.data.uniform_(0,0.05)
        
    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)
    
    def predict(self, user, item):
        return self.forward(user, item)
    
class Loader(Dataset):
    def __init__(self, data):
        self.data = data.copy()
        
        self.pid = self.data.pid.unique()
        self.track_name = self.data.track_name.unique()
        
        # index
        self.pid2idx = {o:i for i,o in enumerate(self.pid)}
        self.track_name2idx = {o:i for i,o in enumerate(self.track_name)}
        
        # reverse index
        self.idx2pid = {i:o for o,i in self.pid2idx.items()}
        self.idx2track_name = {i:o for o,i in self.track_name2idx.items()}
        
        # update data
        self.data.pid = self.data.pid.apply(lambda x: self.pid2idx[x])
        self.data.track_name = self.data.track_name.apply(lambda x: self.track_name2idx[x])
        
        # keep only pid and track_name
        self.x = self.data.values
        self.y = [1 for _ in range(len(self.data))]
        
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y).type(torch.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


model = MatrixFactorization(n_users, n_items, n_factors)
if train_on_gpu:
    model = model.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

playlists = Loader(playlists)
train_size = int(0.8 * len(playlists))
test_size = len(playlists) - train_size

train_set, test_set = random_split(playlists, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

average_train_losses = []
average_test_losses = []
for i in tqdm(range(num_epochs)):
    train_losses = []
    test_losses = []
    
    model.train()
    for x, y in train_loader:
        if train_on_gpu:
            x = x.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        output = model(x[:, 0], x[:, 1])
        loss = criterion(output, y)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            if train_on_gpu:
                x = x.cuda()
                y = y.cuda()
            output = model(x[:, 0], x[:, 1])
            loss = criterion(output, y)
            test_losses.append(loss.item())
    
    # print losses
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_test_loss = sum(test_losses) / len(test_losses)
    average_train_losses.append(avg_train_loss)
    average_test_losses.append(avg_test_loss)
    print(f'epoch: {i:3} train_loss: {avg_train_loss:10.8f} test_loss: {avg_test_loss:10.8f}')
    
    # save the dict of the track_name and track_name embedding to pickle file with epoch number
    track_name_embedding = model.item_factors.weight.data.cpu().numpy()
    track_name_embedding = pd.DataFrame(track_name_embedding)
    track_name_embedding.index = playlists.idx2track_name.values()
    track_name_embedding.to_pickle(f'/data/model/track_name_embedding_{i}.pkl')
    
# find out the best model
best_epoch = np.argmin(average_test_losses)
print(f'best epoch: {best_epoch}, test_loss: {average_test_losses[best_epoch]}')

# remove the other model
for i in range(num_epochs):
    if i != best_epoch:
        os.remove(f'/data/model/track_name_embedding_{i}.pkl')
        
# rename the best model
os.rename(f'/data/model/track_name_embedding_{best_epoch}.pkl', '/data/model/track_name_embedding.pkl')