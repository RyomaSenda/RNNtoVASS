import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from pathlib import Path
# import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, w2v):
        super(NeuralNetwork, self).__init__()
        self.w2v     = w2v
        self.lstm    = nn.LSTM(self.w2v.vec_size,12)
        self.relu    = nn.ReLU()
        self.fc1     = nn.Linear(12, 3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        sentence = x
        x = self.w2v.IDs2Vecs(x)
        x = x.view(len(sentence[0]), 1, -1)
        _,x = self.lstm(x)
        x = self.fc1(x[0].view(-1, 12))
        x = self.softmax(x)
        return x

def train_loop(dataloader, model, loss_fn, optimizer):
    count = 0
    for batch, (X,y) in enumerate(dataloader):
        # 予測と損失の計算
        pred = model(X)
        loss = loss_fn(pred, y)

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_loop(dataloader, model, loss_fn,t):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Epoch {t+1}: Accuracy = {(100*correct):>0.1f}%, Avg loss = {test_loss:>8f}")
