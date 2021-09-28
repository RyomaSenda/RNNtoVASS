import warnings
import sys
import time
import gensim
import logging
import pprint
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.optim as optim

import data_imdb
import W2V
import MLModel

# ハイパーパラメータ
learning_rate = 1e-3
batch_size = 1
epochs_W2V = 1
epochs = 1

def createW2V():
    # Dataset を作成する。(dataset.py参照)
    dict = data_imdb.wordDictionary()
    dataset = data_imdb.MyDataset(dict, None)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # DataLoader を作成する。
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle = True)

    # モデルを作成する(model.py参照)
    model = Word2Vec(dict.len())
    # model = torch.load('model.pth')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("start learning")
    for t in range(epochs_W2V):
        train_loop_W2V(train_dataloader, model, loss_fn, optimizer, dict)
    print("end learning")

    # 学習済みモデルの保存
    torch.save(model.state_dict(), 'W2V_weights.pth')
    torch.save(model, 'W2V.pth')

    return model

def onehot(n,max):
    array = [0] * max
    array[n] = 1
    array = [array]
    return torch.tensor(array, dtype=torch.float)

def insertList(max,neighbors,product,i):
    for j in range(len(max)):
        if product > max[j]:
            max.insert(j, product)
            max.pop()
            neighbors.insert(j, i)
            neighbors.pop()
            break

class Word2Vec(nn.Module):
    def __init__(self, dict_size):
        self.dict_size = dict_size
        self.vec_size = 16
        super(Word2Vec, self).__init__()
        self.w_in    = nn.Linear(self.dict_size,self.vec_size)
        self.w_out   = nn.Linear(self.vec_size,self.dict_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.w_in(x)
        x = self.w_out(x)
        x = self.softmax(x)
        x = x[0]
        return x

    def IDs2Vecs(self,ids):
        x_list = []
        for id in ids[0]:
            x = onehot(id,self.dict_size)
            x = self.w_in(x)
            x_list.append(x)
        return torch.stack(x_list)

    def searchNeighbor(self, dict, id, num):
        x = onehot(id,self.dict_size)
        x = self.w_in(x)
        x = x[0]
        x_norm = torch.matmul(x,x).item() ** 1/2
        neighbors = [0] * num
        max = [0] * num
        words = [""] * num
        for i in range(self.dict_size):
            y = self.w_in(onehot(i,self.dict_size))
            y = y[0]
            y_norm = torch.matmul(y,y).item() ** 1/2
            product = torch.matmul(x,y).item() / (x_norm * y_norm)
            if product > max[num-1]:
                insertList(max,neighbors,product,i)
        for word in dict.dict.keys():
            if dict.dict[word] == id:
                print("Target:")
                print(word)
        for word in dict.dict.keys():
            if dict.dict[word] in neighbors:
                for i in range(num):
                    if dict.dict[word] == neighbors[i]:
                        words[i] = word
                        break
        print(f"Top {num} neighbors:")
        print("ID,\t INNER-PRODUCT,     \t WORD")
        for i in range(num):
            print(f"{neighbors[i]},\t {max[i]},\t {words[i]}")

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
