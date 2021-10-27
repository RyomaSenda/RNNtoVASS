import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from pathlib import Path

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hiddenSize = 512
        self.w2v_vec_size = 300
        self.lstm    = nn.LSTM(self.w2v_vec_size, self.hiddenSize)
        self.fc1     = nn.Linear(self.hiddenSize, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(len(x), 1, -1)
        _,x = self.lstm(x)
        x = self.fc1(x[0].view(-1, self.hiddenSize))
        x = self.softmax(x)
        return x

def train_loop(dataloader, model, loss_fn, optimizer,t):
    count = 0
    test_loss, correct = 0, 0
    max_norm = 10.0
    for batch, (X,y) in enumerate(dataloader):
        # 予測と損失の計算
        count += 1
        if count % 10 == 0: print(f"\r{count}", end = "")
        pred = model(X[0])
        loss = loss_fn(pred, y)

        # training accuracyの計算
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

    size = len(dataloader.dataset)
    test_loss /= size
    correct /= size
    print(f"Epoch {t+1} train: Accuracy = {(100*correct):>0.1f}%, Avg loss = {test_loss:>8f}")

def test_loop(dataloader, model, loss_fn,t):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    PNlist = [0] * 4

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X[0])
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            PNlist[pred.argmax(1)*2 + y] += 1

    test_loss /= size
    correct /= size
    print(f"Epoch {t+1} test : Accuracy = {(100*correct):>0.1f}%, Avg loss = {test_loss:>8f}, PP/PN/NP/NN = {PNlist}")
