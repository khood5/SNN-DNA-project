import torch
import numpy as np
import torch
import torch
from torch.utils.data import DataLoader
import copy
from torch import nn
import multiprocessing as mp
from datetime import datetime

def train(trainData: DataLoader, validData: DataLoader, name: str, featIn: int, return_dict, error_margin, device="cpu"):
  model = nn.Sequential(
            nn.Linear(featIn,700),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(700,700),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(700,700),
            nn.ReLU(),
            nn.Linear(700,1),
          ).to(device)
  MSE = nn.MSELoss(reduction = 'sum')
  adam = torch.optim.Adam(model.parameters(),lr=0.000001,weight_decay=1e-5)
  epochs = 10
  losses = []
  accs = []
  print(f"training {name}...")
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
          totalCorrect = torch.sum(torch.isclose(outputs.int(), targets.int(), atol=error_margin))
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
          totalCorrect = torch.sum(torch.isclose(outputs.int(), targets.int(), atol=error_margin))
          totalCorrect = totalCorrect.clone().detach().cpu().numpy()
          currentAcc.append(float(totalCorrect/len(targets)))
      lastAcc = accs[-1] if accs else -1
      if np.sum(currentAcc)/len(currentAcc) > lastAcc:
        dt_string = datetime.now().strftime("%d.%m.%Y_%H-%M-%S-%f")
        modelPath = f"./Models/smallTrain/{name.replace(' ', '_')}_{dt_string}.pt"
        torch.save(model.state_dict(),modelPath)
        return_dict[name] = {"path":f"{modelPath}", "acc": np.sum(currentAcc)/len(currentAcc)}
        
      accs.append(float(np.sum(currentAcc)/len(currentAcc)))
      losses.append(float(np.sum(avgLoss)/len(avgLoss)))
      
  del model
  torch.cuda.empty_cache()