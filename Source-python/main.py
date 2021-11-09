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
import VASS

# ハイパーパラメータ
learning_rate = 2e-4
batch_size = 1
epochs = 100

args = sys.argv
# 警告文を無視
warnings.simplefilter('ignore')

# ===============================================
# メイン関数
# ===============================================
def main():
    VASS.sampleProgram(args[1])
    return

    w2v = W2V.W2V()
    if args[1] == "learn":
        learn(w2v)
    elif args[1] == "search":
        search(w2v)

# ===============================================
# 関数
# ===============================================
def learn(w2v):
    # Dataset を作成する。(dataset.py参照)
    dataset = data_imdb.MyDataset(w2v, None)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # DataLoader を作成する。
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle = True)

    # モデルを作成する(model.py参照)
    model = MLModel.NeuralNetwork()
    # model = torch.load('model.pth')

    print("Parameters")
    print("-------------------------------")
    print(f"learning_rate = {learning_rate}")
    print(f"   batch_size = {batch_size}")
    print(f"       epochs = {epochs}")
    print(f"    data_size = {len(dataset)}")
    print(f"   train_size = {train_size}")
    print(f"    test_size = {test_size}")
    print("-------------------------------")

    # 学習
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for t in range(epochs):
        MLModel.train_loop(train_dataloader, model, loss_fn, optimizer, t)
        MLModel.test_loop(test_dataloader, model, loss_fn, t)
        if t % 10 == 0:
            torch.save(model.state_dict(), 'model_weights.pth')
            torch.save(model, 'model.pth')
            print("model is saved")

    # 学習済みモデルの保存
    torch.save(model.state_dict(), 'model_weights.pth')
    torch.save(model, 'model.pth')

def search(w2v):
    pos, neg = [], []
    for word in args:
        if word[0] == "+":
            pos.append(word[1:])
        elif word[0] == "-":
            neg.append(word[1:])
    pprint.pprint(w2v.model.most_similar(positive=pos, negative=neg))
    return 0


# =========================================================================
if __name__ == "__main__":
    start = time.time()
    main()
    # print("erapsed_time = {0}".format(time.time() - start) + " [sec]")
