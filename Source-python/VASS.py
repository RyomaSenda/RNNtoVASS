# import torch
# from torch import nn
# from torch.utils.data import DataLoader, Dataset
# from torchvision import datasets, transforms
# from torchvision.transforms import ToTensor, Lambda
# from pathlib import Path
# import torch.optim as optim

class VASS():
    def __init__(self, n):
        super(VASS, self).__init__()
        # self.states   = []
        # self.initial  = 0
        # self.alphabet = []
        # self.trans    = []
        self.acceptvec= []

    def setAcceptvec(dataset):
        for data in dataset:
            print(data)
            break

    def acceptFunc(vec):
        max = 0
        argmax = -1
        for i in range(len(self.acceptvec)):
            if vec * self.acceptvec[i] > max or argmax == -1:
                max = vec * self.acceptvec[i]
                argmax = i
        return argmax
