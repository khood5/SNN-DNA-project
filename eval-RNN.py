import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from dnaDataloader import expermentDataloader
from dnaModelUtil import train
from dnaModelUtil import RNNModel
from dnaDataloader import addData
from dnaDataloader import expermentDataloader
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder = '/home/khood/GitHub/SNN-DNA-project/Prepocessing/sorted/1800_nM_AR_5000'
oneMinInFPS = 1200
batch_size = 10
num_workers = 0
device

def makeDataset(oneTimeUnitInFPS:int, totalRuntime:int, folder:str, batch_size=10, num_workers = 0):
    data = expermentDataloader(
        f"{folder}/index.csv",
        f"{folder}", 
        length = oneTimeUnitInFPS,
    )
    targets = [i[1] for i in data]
    timeSlices = []
    for startTime in range(0, totalRuntime, oneTimeUnitInFPS):
        data = expermentDataloader(
            f"{folder}/index.csv",
            f"{folder}", 
            length = oneTimeUnitInFPS,
            start=startTime
        )
        timeSlices.append([np.array(i[0]) for i in data])
    rawInput = list(zip(*timeSlices))
    rawInput = [np.array(i) for i in rawInput]
    rawData =  list(zip(rawInput,targets))

    trainValidData = []
    testData = []
    addData(testData, trainValidData, rawData, rhsSize=300)


    np.random.shuffle(trainValidData)
    trainData = []
    validData = []
    addData(trainData, validData, trainValidData, rhsSize=int(len(trainValidData)*(1/3)))

    trainDataset = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) 
    validDataset = DataLoader(validData, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testDataset = DataLoader(testData, batch_size=len(testData), shuffle=True, num_workers=num_workers, pin_memory=True)  
    
    return (trainDataset, validDataset, testDataset)

rnn_return_dict = {}
featIn = oneMinInFPS
epochs = 7500
modelSavePath = "./Models/variedMovie/"
for totalRuntime in range(1,31):
      trainDataset, validDataset, testDataset = makeDataset(oneMinInFPS, totalRuntime, folder)
      model = RNNModel(featIn=oneMinInFPS, capacity=int(featIn*0.25), hiddenLayers=4).to(device)
      MSE = nn.MSELoss(reduction = 'mean')
      adam = torch.optim.Adam(model.parameters(),lr=0.00001,weight_decay=1e-5)
      train(trainData=trainDataset, validData=validDataset, name=f"RNN_{totalRuntime}_min", model=model, 
            lossfunction=MSE, optim=adam, return_dict=rnn_return_dict, epochs=epochs,
            device=device, printStatus=True, savePath=modelSavePath)
      
print(rnn_return_dict)