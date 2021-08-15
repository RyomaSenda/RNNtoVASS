import json
import warnings
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.optim as optim

import MyData
import W2V
import MLModel

# ハイパーパラメータ
learning_rate = 1e-3
batch_size = 1
epochs_W2V = 1
epochs = 10

# 警告文を無視 (適宜外してください)
warnings.simplefilter('ignore')


# ===============================================
# メイン関数
# ===============================================
def main():
    w2v = createW2V()
    learn(w2v)

# ===============================================
# 関数
# ===============================================
def createW2V():
    # Dataset を作成する。(dataset.py参照)
    file = MyData.ReadTextfile('finegrained.txt')
    dict = MyData.wordDictionary()
    dataset = MyData.MyDataset(file, dict, None)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # DataLoader を作成する。
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle = True)

    # モデルを作成する(model.py参照)
    model = W2V.Word2Vec(dict.len())
    # model = torch.load('model.pth')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs_W2V):
        W2V.train_loop_W2V(train_dataloader, model, loss_fn, optimizer, dict)

    # 学習済みモデルの保存
    torch.save(model.state_dict(), 'W2V_weights.pth')
    torch.save(model, 'W2V.pth')

    return model

def learn(w2v):
    # Dataset を作成する。(dataset.py参照)
    file = MyData.ReadTextfile('finegrained.txt')
    dict = MyData.wordDictionary()
    dataset = MyData.MyDataset(file, dict, None)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # DataLoader を作成する。
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle = True)

    # モデルを作成する(model.py参照)
    model = MLModel.NeuralNetwork(w2v)
    # model = torch.load('model.pth')

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("Parameters")
    print("-------------------------------")
    print(f"learning_rate = {learning_rate}")
    print(f"   batch_size = {batch_size}")
    print(f"       epochs = {epochs}")
    print(f"    data_size = {len(dataset)}")
    print(f"   proportion = {dataset.proportion}")
    print(f"   train_size = {train_size}")
    print(f"    test_size = {test_size}")
    print("-------------------------------")

    for t in range(epochs):
        MLModel.train_loop(train_dataloader, model, loss_fn, optimizer)
        MLModel.test_loop(test_dataloader, model, loss_fn, t)
    #
    # # 学習済みモデルの保存
    # torch.save(model.state_dict(), 'model_weights.pth')
    # torch.save(model, 'model.pth')


if __name__ == "__main__":
    main()
