import numpy as np
import torch
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader
from torch import nn
from dnaDataloader import expermentDataloader
from datetime import datetime
from dnaDataloader import addData
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
import hashlib


def loadData(path:str):
    print(f"Loading from {path}")
    datasetLarge = expermentDataloader(
        f"{path}/index.csv",
        f"{path}/", 
    )

    rawData = [datasetLarge[i] for i in list(range(len(datasetLarge)))]
    trainData = []
    testValid = []
    addData(trainData, testValid, rawData, rhsSize=int(len(rawData)*0.2))
    testData = []
    validData = []
    addData(testData, validData, testValid, rhsSize=int(len(testValid)*0.6))

    trainDataset = DataLoader(trainData, batch_size=batch_size, shuffle=True)
    testDataset = DataLoader(testData, batch_size=batch_size, shuffle=True)
    validDataset = DataLoader(validData, batch_size=batch_size, shuffle=True)
    return (trainDataset, validDataset, testDataset)

def saveModel(model):
    h = hashlib.sha256()
    h.update((str(datetime.now().timestamp())).encode('utf-8'))
    modelFileName = f"{h.hexdigest()}"
    dt_string = datetime.now().strftime("%d.%m.%Y_%H-%M-%S-%f")
    modelPath = f"./Models/{modelFileName.replace(' ', '_')}_{dt_string}.pt"
    torch.save(model.state_dict(),modelPath)
    print(f"saved to {modelPath}")

def train(model: nn.Sequential, device, trainDataset, validDataset, featIn=12000):
    model.to(device)
    MSE = nn.MSELoss(reduction = 'sum')
    adam = torch.optim.Adam(model.parameters(),lr=0.000001)

    em = 20
    epochs = 100
    losses = []
    accs = []
    print("training...")
    for e in range(epochs): 
        model.train()
        for i, (inputs, targets) in enumerate(trainDataset):
            inputs, targets= inputs.float().to(device), targets.float().to(device)
            outputs = model(inputs)
            loss = MSE(outputs, targets)
            adam.zero_grad()
            loss.backward()
            adam.step()
            
        avgLoss = []
        currentAcc = []
        model.eval()
        for i, (inputs, targets) in enumerate(validDataset):
            inputs, targets= inputs.float().to(device), targets.float().to(device)
            outputs = model(inputs)
            loss = MSE(outputs, targets)
            outputPlot = outputs.clone().detach().cpu().numpy()
            targetsPlot = targets.clone().detach().cpu().numpy()
            avgLoss.append(float(loss.clone().detach().cpu().numpy()))
            totalCorrect = torch.sum(torch.isclose(outputs.int(), targets.int(), atol=em))
            totalCorrect = totalCorrect.clone().detach().cpu().numpy()
            currentAcc.append(float(totalCorrect/len(targets)))
            print(f"\
            epoch: {e}/{epochs}\t \
            loss:{np.sum(avgLoss)/len(avgLoss)}\t \
            accuracy:{np.sum(currentAcc)/len(currentAcc)} \
            ",end="\x1b\r")
        accs.append(float(np.sum(currentAcc)/len(currentAcc)))
        losses.append(float(np.sum(avgLoss)/len(avgLoss)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''runs training for MLP model \n''', 
                                    formatter_class=RawTextHelpFormatter )
    parser.add_argument('device', help='device to run training on ex: cuda:2, cuda:0, cpu\n',
                        type=str)
    parser.add_argument('name', help='name for model\n',
                        type=str)
    parser.add_argument('data', help='root dir for data (without trailing /)\n',
                        type=str)
    args = parser.parse_args()
    
    

    device = torch.device(args.device)
    batch_size = 10

    featIn = 12000
    model = nn.Sequential(
          nn.Linear(featIn,24000),
          nn.ReLU(),
          nn.Linear(24000,24000),
          nn.ReLU(),
          nn.Linear(24000,24000),
          nn.ReLU(),
          nn.Linear(24000,1),
        )
    model.to(device)
    trainDataset, validDataset, testDataset = loadData(args.data)
    train(model, device, trainDataset, validDataset)
    saveModel(model)