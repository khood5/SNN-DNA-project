import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dnaDataloader import expermentDataloader
from dnaDataloader import addData
from dnaModelUtil import MLPModel
from dnaModelUtil import train
import json

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
folder = '/home/khood/GitHub/SNN-DNA-project/Prepocessing/sorted/1800_nM_AR_5000'
oneMinInFPS = 1200
batch_size = 10
num_workers = 0
device

mlp_return_dict = {}
epochs = 1000
modelSavePath = "./Models/variedMovie/"
for length in range(200, int(oneMinInFPS*5), 200):
    print(f"Trying {length} frames ...")
    data = expermentDataloader(
        f"{folder}/index.csv",
        f"{folder}", 
        length=length
    )
    rawData = [d for d in data]
    featIn = len(rawData[0][0])
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

    model = MLPModel(featIn=featIn, capacity=int(400)).to(device)
    MSE = nn.MSELoss(reduction = 'mean')
    adam = torch.optim.Adam(model.parameters(),lr=0.00001,weight_decay=1e-5)

    train(trainData=trainDataset, validData=validDataset, name=f"MLP_{length}_frames", model=model,
          lossfunction=MSE, optim=adam, return_dict=mlp_return_dict, epochs=epochs, 
          device=device, savePath=modelSavePath)

with open(f'{modelSavePath}mlp_valid_acc.json', 'w') as convert_file:
     convert_file.write(json.dumps(mlp_return_dict))
print(mlp_return_dict)