import torch
import numpy as np
import torch
import torch
from torch.utils.data import DataLoader
from torch import nn
from scipy import stats as st
import copy
import os

def calc_margin_of_error(targets: np.ndarray):
  σ = np.std(targets)
  n = len(targets)
  z = st.zscore(targets)
  return z * (σ/np.sqrt(n))

def train(trainData: DataLoader, validData: DataLoader, name: str, savePath: str, model, lossfunction, optim, epochs: int, return_dict={}, margin_of_error=20, device=torch.device("cpu"), printStatus=False):
  model.to(device)
  losses = []
  accs = []
  if printStatus:
    print(f"training {name} on {device}...")
  for e in range(epochs):
    avgLossTrain = []
    currentAccTrain = [] 
    model.train()
    for i, (inputs, targets) in enumerate(trainData):
        inputs, targets= inputs.float().to(device), targets.float().to(device)
        outputs = model(inputs)
        loss = lossfunction(outputs, targets)
        avgLossTrain.append(float(loss.item()))
        optim.zero_grad()
        loss.backward()
        optim.step()
        totalCorrect = torch.sum(torch.isclose(outputs.int(), targets.int(), atol=margin_of_error))
        totalCorrect = totalCorrect.item()
        currentAccTrain.append(float(totalCorrect/len(targets)))
        
    avgLoss = []
    currentAcc = []
    model.eval()
    for i, (inputs, targets) in enumerate(validData):
        inputs, targets= inputs.float().to(device), targets.float().to(device)
        outputs = model(inputs)
        loss = lossfunction(outputs, targets)
        avgLoss.append(float(loss.clone().detach().cpu().numpy()))
        totalCorrect = torch.sum(torch.isclose(outputs.int(), targets.int(), atol=margin_of_error))
        totalCorrect = totalCorrect.clone().detach().cpu().numpy()
        currentAcc.append(float(totalCorrect/len(targets)))
        if printStatus:
          print(f"\
          epoch: {e}/{epochs}\t \
          Train Loss:{'%.4f' % (np.sum(avgLossTrain)/len(avgLossTrain))} Valid Loss:{'%.4f' % (np.sum(avgLoss)/len(avgLoss))}\t \
          Train accuracy:{'%.4f' % (np.sum(currentAccTrain)/len(currentAccTrain))} Valid accuracy:{'%.4f' % (np.sum(currentAcc)/len(currentAcc))} \
          ",end="\x1b\r")
    lastAcc = accs[-1] if accs else -1
    if np.sum(currentAcc)/len(currentAcc) > lastAcc:
      modelPath = os.path.join(savePath,f"{name.replace(' ', '_')}.pt")
      torch.save(model.state_dict(),modelPath)
      return_dict[name] = {"path":f"{modelPath}", "acc": np.sum(currentAcc)/len(currentAcc)}
    accs.append(float(np.sum(currentAcc)/len(currentAcc)))
    losses.append(float(np.sum(avgLoss)/len(avgLoss)))
  del model
  torch.cuda.empty_cache()
  return {"loss_val": losses, "accuracy_val":accs }
  
def test(testData: DataLoader, modelPath: str, name: str, model, return_dict: dict, epochs: int, margin_of_error=20, device=torch.device("cpu")):
  model.to(device)
  model.load_state_dict(torch.load(modelPath))
  model.to(device)
  model.eval()
  print(f"test {name} on {device}...")
  for _ in range(epochs): 
    for _, (inputs, targets) in enumerate(testData):
        inputs, targets= inputs.float().to(device), targets.float().to(device)
        outputs = model(inputs)
        outputPlot = outputs.clone().detach().cpu().numpy()
        targetPlot = targets.clone().detach().cpu().numpy()
        totalCorrect = torch.sum(torch.isclose(outputs.int(), targets.int(), atol=margin_of_error))
        totalCorrect = totalCorrect.clone().detach().cpu().numpy()
        return_dict[name] = {"outputPlot":outputPlot, "targetPlot": targetPlot, "acc":float(totalCorrect/len(targets))}

def averageDiff(data: list):
    diffs = []
    while data:
        point = data.pop()
        for i in data:
            diffs.append(abs(point - i))
    return np.sum(diffs)/len(diffs)
  
def printStats(data: list, name="", other=[]):
    data = copy.deepcopy(data)
    print(f"+---------- {name} ----------")
    print(f"| total number of experiments: {len(data)}")
    print(f"| min: {np.min(data)}")
    print(f"| max: {np.max(data)}")
    print(f"| average: {np.average(data)}")
    print(f"| median: {np.median(data)}")
    print(f"| mode: {st.mode(data, keepdims=False)}")
    print(f"| std: {np.std(data)}")
    print(f"| average difference: {averageDiff(data)}")
    if other:
      for o in other:
        print(f"| {o}")
    print("+------------------")
    
class RNNModel(nn.Module):
  def __init__(self, featIn, capacity, hiddenLayers=2, device=torch.device("cpu")):
      super(RNNModel, self).__init__()
      self.capacity = capacity
      self.featIn = featIn
      self.hiddenLayers = hiddenLayers
      self.device = device
      # RNN
      self.rnn = nn.RNN(featIn, self.capacity, hiddenLayers, batch_first=True, nonlinearity='relu', dropout=0.2)
      
      # Readout layer
      self.fc = nn.Linear(capacity, 1)
  
  def forward(self, x):
      # Initialize hidden state with zeros
      h0 = torch.zeros(self.hiddenLayers, x.size(0),self.capacity).to(x.get_device())
      out, hn = self.rnn(x, h0)
      out = self.fc(out[:, -1, :]) 
      return out
    
class MLPModel(nn.Module):
    def __init__(self, featIn, capacity):
        super().__init__()
        self.input = nn.Linear(featIn,capacity)
        self.inputAct = nn.Tanh()
        self.inputDropout = nn.Dropout(p=0.2)
        self.hidden1 = nn.Linear(capacity,capacity)
        self.hidden1Act = nn.Tanh()
        self.hidden1Dropout = nn.Dropout(p=0.2)
        self.hidden2 = nn.Linear(capacity,capacity)
        self.hidden2Act = nn.Tanh()
        self.output = nn.Linear(capacity,1)
        
    def forward(self, x):
        x = self.inputAct(self.input(x))
        x = self.inputDropout(x)
        x = self.hidden1Act(self.hidden1(x))
        x = self.hidden1Dropout(x)
        x = self.hidden2Act(self.hidden2(x))
        x = self.output(x)
        return x
      
# progress_bar for when stuff takes a while to load
def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)