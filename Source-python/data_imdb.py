import json
import glob
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from pathlib import Path

import W2V

# txtファイルを受け取り，成形済みデータセットを返す．
# 単語 to ID の辞書をもち，1データは (文章をID系列に変換したもの, -1 or 0 or 1) の組
class MyDataset(Dataset):
    def __init__(self, w2v, transform=None):
        self.sentences, self.labels = self._get_sentences_()
        self.w2v = w2v

    def __getitem__(self, index):
        # label = [0,0,0]
        # label[self.labels[index] + 1] = 1
        # return (torch.tensor(self.sentences[index]), torch.tensor(label, dtype=torch.long))
        return (self.w2v.words2Vecs(self.sentences[index]), self.labels[index])

    def _get_sentences_(self):
        sentences, labels = [], []
        filepath = "D:/Dataset/imdb/aclImdb/train"
        for b in [0,1]:
            if b == 0: fileList = glob.glob(filepath + "/pos/*")
            else: fileList = glob.glob(filepath + "/neg/*")
            for filename in fileList:
                _sentence = self.readSentence(filename)
                sentences.append(_sentence)
                labels.append(b)
        return sentences, labels

    def readSentence(self,filename):
        file = open(filename, encoding='UTF-8')
        specialCharacter = ".,!?;:\'\"\n()<>/"
        s = file.read()
        for sc in specialCharacter:
            s = s.replace(sc,'')
        s = s.lower()
        return s.split(" ")

    def __len__(self):
        return len(self.sentences)

# 単語辞書 dict: word(string) -> ID(int)
# class wordDictionary():
#     def __init__(self):
#         self.dict = {}
#
#     def setword(self, word):
#         if word not in self.dict.keys():
#             self.dict[word] = len(self.dict)
#
#     def getID(self,word):
#         return self.dict[word]
#
#     def len(self):
#         return len(self.dict)
