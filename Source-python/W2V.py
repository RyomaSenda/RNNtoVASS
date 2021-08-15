import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from pathlib import Path
# import torch.optim as optim

def onehot(n,max):
    array = [0] * max
    array[n] = 1
    return torch.tensor(array, dtype=torch.float)

class Word2Vec(nn.Module):
    def __init__(self, dict_size):
        self.dict_size = dict_size
        self.vec_size = 16
        super(Word2Vec, self).__init__()
        self.w_in    = nn.Linear(self.dict_size,self.vec_size)
        self.w_out   = nn.Linear(self.vec_size,self.dict_size)
        # self.relu    = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.w_in(x)
        x = self.w_out(x)
        # x = self.softmax(x)
        return x

    def IDs2Vecs(self,ids):
        x_list = []
        for id in ids[0]:
            x = onehot(id,self.dict_size)
            x = self.w_in(x)
            x_list.append(x)
        return torch.stack(x_list)

def train_loop_W2V(dataloader, model, loss_fn, optimizer, dict):
    for batch, (X,y) in enumerate(dataloader):
        for x in X[0]:
            x = onehot(x,dict.len())
            # 予測と損失の計算
            pred = model(x)
            pred = [pred] * len(X[0])
            pred = torch.stack(pred)
            loss = loss_fn(pred, X[0])

            # バックプロパゲーション
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
