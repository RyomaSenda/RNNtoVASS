# import torch
# from torch import nn
# from torch.utils.data import DataLoader, Dataset
# from torchvision import datasets, transforms
# from torchvision.transforms import ToTensor, Lambda
# from pathlib import Path
# import torch.optim as optim

CVecLen = 2

# ===============================================
# メイン関数
# ===============================================

def sampleProgram(word):
    vass = VASS()
    vass.loadVASS("vass.txt")
    # for i in range(2):
    #     vass.addState()
    # for i in range(2):
    #     vass.addSymbol()
    #
    states = vass.states
    symbols = vass.symbols
    # vass.addTransition(states[0],symbols[0],states[1],[2,1])
    # vass.addTransition(states[1],symbols[1],states[1],[-1,0])
    # vass.addTransition(states[1],symbols[0],states[2],[2,2])
    #
    outputs = vass.outputs
    # vass.addClassifyCondition(states[2],outputs[0],[3,1])
    # vass.addClassifyCondition(states[2],outputs[1],[-3,3])

    input = []
    for s in word:
        if s == "a": input.append(symbols[0])
        elif s == "b": input.append(symbols[1])

    print("================================")
    output = vass.run(input)
    print(f"Output is {output.id}")
    print("================================")
    vass.saveVASS("vass.txt")
    return

class State():
    def __init__(self, id):
        super(State, self).__init__()
        self.id = id

class Symbol():
    def __init__(self, id):
        super(Symbol, self).__init__()
        self.id = id

class Output():
    def __init__(self, id):
        super(Output, self).__init__()
        self.id = id

class Transition():
    def __init__(self, preState, symbol, nextState, vector):
        super(Transition, self).__init__()
        self.preState  = preState
        self.symbol    = symbol
        self.nextState = nextState
        self.vector    = vector

class ClassifyCondition():
    def __init__(self, state, output, vector):
        super(ClassifyCondition, self).__init__()
        self.state  = state
        self.output = output
        self.vector = vector

class Configuration():
    def __init__(self, state, vector):
        super(Configuration, self).__init__()
        self.state  = state
        self.vector = vector

class VASS():
    def __init__(self):
        super(VASS, self).__init__()
        # VASSの定義
        self.iniState           = State(-1)
        self.states             = []
        self.symbols            = []
        self.transitions        = []
        self.classifyConditions = []
        self.outputs            = [Output(0),Output(1)]

#=============================================================================
# VASSを構築する．
#=============================================================================
    def addState(self):
        self.states.append(State(len(self.states)))

    def addSymbol(self):
        self.symbols.append(Symbol(len(self.symbols)))

    def addTransition(self, preState, symbol, nextState, vector):
        self.transitions.append(Transition(preState, symbol, nextState, vector))

    def addClassifyCondition(self, state, output, vector):
        self.classifyConditions.append(ClassifyCondition(state, output, vector))

    # def addOutput(self):
    #     self.outputs.append(Output(len(self.outputs)))

#=============================================================================
# VASSを実行する．
#=============================================================================
    def run(self, sequence):
        conf = self.initConfiguration()
        for symbol in sequence:
            conf = self.transit(conf, symbol)
        return self.getOutput(conf)

    def initConfiguration(self):
        return Configuration(self.iniState, [0] * CVecLen)

    def transit(self, conf, symbol):
        nextState, vector = self.transitionFunction(conf.state, symbol)
        conf.state  = nextState
        conf.vector = [conf.vector[0] + vector[0],conf.vector[1] + vector[1]]
        return conf

    def transitionFunction(self, preState, symbol):
        for t in self.transitions:
            if t.preState == preState and t.symbol == symbol:
                return t.nextState, t.vector
        print("Transition is stopped.")
        exit()

    def getOutput(self, conf):
        minDistance = -1
        minOutput   = Output(-1)
        for output in self.outputs:
            _vector = self.getClassifyVector(conf.state, output)
            if _vector is not None:
                dist = euclideanDistance(conf.vector, _vector)
                if minDistance == -1 or minDistance > dist:
                    minDistance = dist
                    minOutput = output
        if minDistance == -1:
            print("There is no valid output.")
            exit()
        return minOutput

    def getClassifyVector(self, state, output):
        for cond in self.classifyConditions:
            if cond.state.id == state.id and cond.output.id == output.id:
                return cond.vector
        return None

#=============================================================================
# VASSをセーブ/ロードする．
#=============================================================================
    def saveVASS(self, filename):
        f = open(filename, "w")
        for state in self.states:
            f.write(f"{state.id}=")
        f.write("\n")
        for symbol in self.symbols:
            f.write(f"{symbol.id}=")
        f.write("\n")
        for t in self.transitions:
            f.write(f"{t.preState.id}_{t.symbol.id}_{t.nextState.id}_{t.vector}=")
        f.write("\n")
        for cc in self.classifyConditions:
            f.write(f"{cc.state.id}_{cc.output.id}_{cc.vector}=")
        f.write("\n")
        for output in self.outputs:
            f.write(f"{output.id}=")
        f.close()

    def loadVASS(self, filename):
        f = open(filename, "r")
        datalist = f.readlines()
        stateIds = datalist[0].split("=")[:-1]
        symbolIds = datalist[1].split("=")[:-1]
        tTuples = datalist[2].split("=")[:-1]
        ccTuples = datalist[3].split("=")[:-1]
        outputIds = datalist[4].split("=")[:-1]
        self.states = []
        for i in stateIds:
            self.states.append(State(int(i)))
        self.iniState = self.states[0]
        for i in symbolIds:
            self.symbols.append(Symbol(int(i)))
        for tTuple in tTuples:
            tList = tTuple.split("_")
            tList[3] = tList[3].replace("[","").replace("]","").replace(" ","")
            tList[3] = tList[3].split(",")
            vector = []
            for string in tList[3]:
                vector.append(float(string))
            self.transitions.append(Transition(self.states[int(tList[0])], self.symbols[int(tList[1])], self.states[int(tList[2])], vector))
        for ccTuple in ccTuples:
            ccList = ccTuple.split("_")
            ccList[2] = ccList[2].replace("[","").replace("]","").replace(" ","")
            ccList[2] = ccList[2].split(",")
            vector = []
            for string in ccList[2]:
                vector.append(float(string))
            self.classifyConditions.append(ClassifyCondition(self.states[int(ccList[0])], self.outputs[int(ccList[1])], vector))
        self.outputs = []
        for i in outputIds:
            self.outputs.append(Output(int(i)))

###############################################################################
def euclideanDistance(vector1, vector2):
    dist = 0
    for i in range(len(vector1)):
        dist += (vector1[i] - vector2[i]) ** 2
    dist = dist ** 0.5
    return dist
