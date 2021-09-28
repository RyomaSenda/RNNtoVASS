import gensim
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.optim as optim

class W2V():
    def __init__(self):
        self.model = self.createW2V()
        self.vec_size = 300

    def createW2V(self):
        filename = 'D:/Dataset/GoogleNews-vectors-negative300.bin'
        model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
        return model

    def words2Vecs(self,sentence):
        vec_list = []
        for word in sentence:
            if word in self.model:
                array = self.model[word]
                list = []
                for num in array:
                    list.append(num)
                vec_list.append(torch.tensor(list))
            else:
                vec_list.append(torch.tensor([0] * self.vec_size))
        return torch.stack(vec_list)
