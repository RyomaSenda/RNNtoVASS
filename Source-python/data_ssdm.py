import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from pathlib import Path

# txtファイル読み取りを扱うクラス
class ReadTextfile():
    def __init__(self, filename):
        self.file = open(filename, encoding='UTF-8')

    # return: (string, int)
    # EOFの場合は ("", -2)
    def getitem(self):
        tag = {"neg": -1, "neu": 0, "pos": 1}
        specialCharacter = ".,!?;:\"\n()"
        while(True):
            s = self.file.readline()
            if len(s) >= 3:
                if s[0:3] in tag.keys():
                    for sc in specialCharacter:
                        s = s.replace(sc,'')
                    s = s.lower()
                    return s[4:].split(" "), tag[s[0:3]]
            if not s:
                return "", -2

# txtファイルを受け取り，成形済みデータセットを返す．
# 単語 to ID の辞書をもち，1データは (文章をID系列に変換したもの, -1 or 0 or 1) の組
class MyDataset(Dataset):
    def __init__(self, dict, transform=None):
        file = ReadTextfile('finegrained.txt')
        self.sentences, self.labels, self.proportion = self._get_sentences_(file, dict)

    def __getitem__(self, index):
        # label = [0,0,0]
        # label[self.labels[index] + 1] = 1
        # return (torch.tensor(self.sentences[index]), torch.tensor(label, dtype=torch.long))
        return (torch.tensor(self.sentences[index]), self.labels[index] + 1)

    def _get_sentences_(self, file, dict):
        sentences = []
        labels = []
        proportion = [0,0,0]
        while(True):
            sentence = []
            _sentence,_tag = file.getitem()
            if _tag == -2: break
            for word in _sentence:
                dict.setword(word)
                sentence.append(dict.getID(word))
            sentences.append(sentence)
            labels.append(_tag)
            proportion[_tag + 1] += 1
        return sentences, labels, proportion

    def __len__(self):
        return len(self.sentences)

# 単語辞書 dict: word(string) -> ID(int)
class wordDictionary():
    def __init__(self):
        self.dict = {}

    def setword(self, word):
        if word not in self.dict.keys():
            self.dict[word] = len(self.dict)

    def getID(self,word):
        return self.dict[word]

    def len(self):
        return len(self.dict)
