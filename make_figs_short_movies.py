import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch
from torch import nn
from dnaDataloader import expermentDataloader
from dnaDataloader import addData
from dnaModelUtil import MLPModel
from dnaModelUtil import train
from dnaModelUtil import progress_bar
from datetime import datetime
import json
import pandas as pd
import seaborn as sns
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
folder = '/home/khood/GitHub/SNN-DNA-project/Prepocessing/sorted/1800_nM_AR_5000'
oneMinInFPS = 1200
batch_size = 10
num_workers = 0

def makeDatasets(length: int, folder: str):
    length = int(length*oneMinInFPS)
    assert length > 0
    data = expermentDataloader(
        f"{folder}/index.csv",
        f"{folder}",
        length=length
    )
    rawData = [d for d in data]
    trainValidData = []
    testData = []
    addData(testData, trainValidData, rawData, rhsSize=300)

    np.random.shuffle(trainValidData)
    trainData = []
    validData = []
    addData(trainData, validData, trainValidData,
            rhsSize=int(len(trainValidData)*(1/3)))

    trainDataset = DataLoader(trainData, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    validDataset = DataLoader(validData, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    testDataset = DataLoader(testData, batch_size=len(
        testData), shuffle=True, num_workers=num_workers, pin_memory=True)
    return trainDataset, validDataset, testDataset

def do_training():
    model_dir = "/home/khood/GitHub/SNN-DNA-project/Models/modelsForVariedLengthed"
    resultes = {}
    return_dict = {}
    current = 0
    keys = list(datasests.keys())
    while keys:
        length = float(keys.pop(0))
        val_data = datasests[f"{length}"][1]
        train_data = datasests[f"{length}"][0]
        # next(iter(train_data))
        featIn = len(next(iter(train_data))[0][0])
        # print(f"featIn: {featIn}")
        model = MLPModel(featIn=int(featIn), capacity=1000)
        model = model.to(device)
        MSE = nn.MSELoss(reduction='mean')
        adam = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
        resultes[f"{length}"] = train(trainData=train_data, validData=val_data, name=f"{length}_min(s)", savePath=model_dir, margin_of_error=5,
                                    model=model, lossfunction=MSE, optim=adam, return_dict=return_dict, epochs=1000, device=device, printStatus=False)
        current += 1
        progress_bar(current=current, total=total, bar_length=20)
    return resultes



if __name__ == "__main__":
    datasests = {}
    maxLength = 3*oneMinInFPS
    minLength = 1/oneMinInFPS
    step = (1/oneMinInFPS)*100000
    current = 0
    total = len(np.arange(minLength, maxLength+step, step))
    print("making data...")
    for length in np.arange(minLength, maxLength+step, step):
        datasests[f"{length}"] = makeDatasets(length, folder)
        current += 1
        progress_bar(current=current, total=total, bar_length=20)
    print("running training...")
    results = do_training()
    with open('/home/khood/GitHub/SNN-DNA-project/Models/mlp_1frame_1min_acc.json', "w") as f:
        json.dump(results, f)
    header = []
    acc_data = []
    loss_data = []
    for min in results.keys():
        header.append(min)
        acc_data.append(results[min]["accuracy_val"])
        loss_data.append(results[min]["loss_val"])

    acc_data_df = pd.DataFrame(acc_data, index=header).T
    loss_data = pd.DataFrame(loss_data, index=header).T
    acc_data_df.to_csv("/home/khood/GitHub/SNN-DNA-project/acc_data.csv",index=False)
    loss_data.to_csv("/home/khood/GitHub/SNN-DNA-project/loss_data.csv",index=False)