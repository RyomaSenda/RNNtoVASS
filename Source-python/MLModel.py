import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from pathlib import Path

import VASS
import converter

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hiddenSize = 128
        self.w2v_vec_size = 300
        self.lstm    = nn.LSTM(self.w2v_vec_size, self.hiddenSize)
        self.fc1     = nn.Linear(self.hiddenSize, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        self.converter = converter.Converter(8, 4, self.w2v_vec_size, self.hiddenSize, 2)

    def forward(self, x):
        x = x.view(len(x), 1, -1)
        _,x = self.lstm(x)
        x = self.fc1(x[0].view(-1, self.hiddenSize))
        x = self.softmax(x)
        return x

    def extract(self, x):
        h,c = torch.zeros([1,1,self.hiddenSize]), torch.zeros([1,1,self.hiddenSize])
        x = x.view(len(x), 1, -1)
        for i in range(len(x)):
            xi = torch.zeros([1,1,300])
            xi[0][0] = x[i][0]
            _, (hh,cc) = self.lstm(xi,(h,c))
            self.converter.addTransitionHistory(tensorToList(h[0][0]), tensorToList(x[0][0]), tensorToList(hh[0][0]), listDif(tensorToList(cc[0][0]),tensorToList(c[0][0])))
            h,c = hh,cc
        x = self.fc1(h.view(-1, self.hiddenSize))
        x = self.softmax(x)

        self.converter.addCcHistory(tensorToList(h[0][0]),x.argmax(1).item(),tensorToList(c[0][0]))
        return x

def tensorToList(tensor):
    list = []
    for i in range(len(tensor)):
        list.append(tensor[i].item())
    return list

def listDif(list1, list2):
    for i in range(len(list1)):
        list1[i] -= list2[i]
    return list1

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

def vass_loop(dataloader, model, vass):
    with torch.no_grad():
        count = 0
        for X, y in dataloader:
            pred = model.extract(X[0])
            count += 1
            if count % 10 == 0:
                print(f"\r{count}", end = "")
    for i in range(model.converter.stateDimension): vass.addState()
    for i in range(model.converter.inputDimension): vass.addSymbol()
    vass.iniState = vass.states[0]
    transitions = model.converter.getTransitions()
    for i in range(model.converter.inputDimension):
        for j in range(model.converter.stateDimension):
            vass.addTransition(vass.states[j],vass.symbols[i],vass.states[transitions[i][j][0]], transitions[i][j][1])
    ccs = model.converter.getCcs()
    for i in range(model.converter.stateDimension):
        for j in range(model.converter.outputMax):
            vass.addClassifyCondition(vass.states[i],vass.outputs[j],ccs[i][j])
    return vass
