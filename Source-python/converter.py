# import torch
import math
# from torch import nn
# from torch.utils.data import DataLoader, Dataset
# from torchvision import datasets, transforms
# from torchvision.transforms import ToTensor, Lambda
# from pathlib import Path
# import torch.optim as optim

import VASS

class TransitionHistory():
    def __init__(self, preState, input, nextStateVec, cVec):
        super(TransitionHistory, self).__init__()
        self.preState = preState
        self.input = input
        self.nextStateVec = nextStateVec
        self.cVec = cVec

class CcHistory():
    def __init__(self, state, output, cVec):
        super(CcHistory, self).__init__()
        self.state = state
        self.output = output
        self.cVec = cVec

class Converter():
    def __init__(self, inputDiv, stateDiv, inputMax, stateMax, outputMax):
        super(Converter, self).__init__()
        self.inputDiv = inputDiv
        self.stateDiv = stateDiv
        self.inputDimension = 2 ** inputDiv
        self.stateDimension = 2 ** stateDiv
        self.inputMax = inputMax
        self.stateMax = stateMax
        self.outputMax = outputMax
        self.transitionHistory = []
        self.ccHistory = []

    def inputVec2Symbol(self, vector):
        if self.inputDiv == 0: return 0
        inputId = 0
        for i in range(self.inputDiv):
            spanTotal = 0
            for j in range(math.floor(i*self.inputMax/self.inputDiv), math.floor((i+1)*self.inputMax/self.inputDiv)):
                spanTotal += vector[j]
            if spanTotal >= 0:
                inputId += 2 ** i
        return inputId

    def stateVec2Symbol(self, vector):
        if self.stateDiv == 0: return 0
        stateId = 0
        for i in range(self.stateDiv):
            spanTotal = 0
            for j in range(math.floor(i*self.stateMax/self.stateDiv), math.floor((i+1)*self.stateMax/self.stateDiv)):
                spanTotal += vector[j]
            if spanTotal >= 0:
                stateId += 2 ** i
        return stateId

    def addTransitionHistory(self, preStateVec, inputVec, nextStateVec, cVec):
        input = self.inputVec2Symbol(inputVec)
        preState = self.stateVec2Symbol(preStateVec)
        self.transitionHistory.append(TransitionHistory(preState,input,nextStateVec,cVec))

    def getTransitions(self):
        transitions = [[[0,0,0] for j in range(self.stateDimension)] for i in range(self.inputDimension)]
        for t in self.transitionHistory:
            if transitions[t.input][t.preState][2] == 0:
                transitions[t.input][t.preState][0] = t.nextStateVec
                transitions[t.input][t.preState][1] = t.cVec
                transitions[t.input][t.preState][2] = 1
            else:
                NSV = []
                for i in range(len(t.nextStateVec)):
                    NSV.append(transitions[t.input][t.preState][0][i] + t.nextStateVec[i])
                CV = []
                for i in range(len(t.cVec)):
                    CV.append(transitions[t.input][t.preState][1][i] + t.cVec[i])
                transitions[t.input][t.preState][0] = NSV
                transitions[t.input][t.preState][1] = CV
                transitions[t.input][t.preState][2] += 1
        for i in range(self.inputDimension):
            for j in range(self.stateDimension):
                if transitions[i][j][2] != 0:
                    NSV = []
                    for k in range(len(transitions[i][j][0])):
                        NSV.append(transitions[i][j][0][k] / transitions[i][j][2])
                    CV = []
                    for k in range(len(transitions[i][j][1])):
                        CV.append(transitions[i][j][1][k] / transitions[i][j][2])
                    transitions[i][j] = (self.stateVec2Symbol(NSV), CV)
                else:
                    transitions[i][j] = (j, [0]*self.stateMax)
        return transitions

    def addCcHistory(self, stateVec, output, cVec):
        state = self.stateVec2Symbol(stateVec)
        self.ccHistory.append(CcHistory(state,output,cVec))

    def getCcs(self):
        ccs = [[[0,0] for j in range(self.outputMax)] for i in range(self.stateDimension)]
        for cc in self.ccHistory:
            if ccs[cc.state][cc.output][1] == 0:
                ccs[cc.state][cc.output][0] = cc.cVec
                ccs[cc.state][cc.output][1] = 1
            else:
                CV = []
                for i in range(len(cc.cVec)):
                    CV.append(ccs[cc.state][cc.output][0][i] + cc.cVec[i])
                ccs[cc.state][cc.output][0] = CV
                ccs[cc.state][cc.output][1] += 1
        for i in range(self.stateDimension):
            for j in range(self.outputMax):
                if ccs[i][j][1] != 0:
                    CV = []
                    for k in range(len(ccs[i][j][0])):
                        CV.append(ccs[i][j][0][k] / ccs[i][j][1])
                    ccs[i][j] = CV
                else:
                    ccs[i][j] = [0]*self.stateMax
        return ccs
