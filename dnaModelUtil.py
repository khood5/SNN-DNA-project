import torch
import numpy as np
import torch
import torch
from torch.utils.data import DataLoader
from torch import nn
from scipy import stats as st
import copy

def calc_margin_of_error(targets: np.array):
  σ = np.std(targets)
  n = len(targets)
  z = st.zscore(targets)
  return z * (σ/np.sqrt(n))

def train(trainData: DataLoader, validData: DataLoader, name: str, featIn: int, return_dict, epochs, margin_of_error, device=torch.device("cpu"), capacity=700, printStatus=False):
  model = nn.Sequential(
            nn.Linear(featIn,capacity),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(capacity,capacity),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(capacity,capacity),
            nn.ReLU(),
            nn.Linear(capacity,1),
          ).to(device)
  MSE = nn.MSELoss(reduction = 'sum')
  adam = torch.optim.Adam(model.parameters(),lr=0.000001,weight_decay=1e-5)
  losses = []
  accs = []
  print(f"training {name} on {device}...")
  for e in range(epochs):
    avgLossTrain = []
    currentAccTrain = [] 
    model.train()
    for i, (inputs, targets) in enumerate(trainData):
        inputs, targets= inputs.float().to(device), targets.float().to(device)
        outputs = model(inputs)
        loss = MSE(outputs, targets)
        avgLossTrain.append(float(loss.item()))
        adam.zero_grad()
        loss.backward()
        adam.step()
        totalCorrect = torch.sum(torch.isclose(outputs.int(), targets.int(), atol=em))
        totalCorrect = totalCorrect.item()
        currentAccTrain.append(float(totalCorrect/len(targets)))
        
    avgLoss = []
    currentAcc = []
    model.eval()
    for i, (inputs, targets) in enumerate(validData):
        inputs, targets= inputs.float().to(device), targets.float().to(device)
        outputs = model(inputs)
        loss = MSE(outputs, targets)
        avgLoss.append(float(loss.clone().detach().cpu().numpy()))
        totalCorrect = torch.sum(torch.isclose(outputs.int(), targets.int(), atol=margin_of_error))
        totalCorrect = totalCorrect.clone().detach().cpu().numpy()
        currentAcc.append(float(totalCorrect/len(targets)))
        if not printStatus:
          print(f"\
          epoch: {e}/{epochs}\t \
          Train Loss:{'%.4f' % (np.sum(avgLossTrain)/len(avgLossTrain))} Valid Loss:{'%.4f' % (np.sum(avgLoss)/len(avgLoss))}\t \
          Train accuracy:{'%.4f' % (np.sum(currentAccTrain)/len(currentAccTrain))} Valid accuracy:{'%.4f' % (np.sum(currentAcc)/len(currentAcc))} \
          ",end="\x1b\r")
    lastAcc = accs[-1] if accs else -1
    if np.sum(currentAcc)/len(currentAcc) > lastAcc:
      modelPath = f"./Models/smallTrain/{name.replace(' ', '_')}.pt"
      torch.save(model.state_dict(),modelPath)
      return_dict[name] = {"path":f"{modelPath}", "acc": np.sum(currentAcc)/len(currentAcc)}
    accs.append(float(np.sum(currentAcc)/len(currentAcc)))
    losses.append(float(np.sum(avgLoss)/len(avgLoss)))
  del model
  torch.cuda.empty_cache()
  
def test(testData: DataLoader, modelPath: str, name: str, featIn: int, return_dict, epochs, margin_of_error, device=torch.device("cpu"), capacity=700):
  model = nn.Sequential(
            nn.Linear(featIn,capacity),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(capacity,capacity),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(capacity,capacity),
            nn.ReLU(),
            nn.Linear(capacity,1),
          ).to(device)
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