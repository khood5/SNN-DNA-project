import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from dnaDataloader import expermentDataloader
from dnaModelUtil import train, test
from dnaModelUtil import RNNModel
from dnaDataloader import addData
from dnaDataloader import expermentDataloader
from torch.utils.data import DataLoader
import json
import os

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


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder = '/users/kent/student/khood5/GitHub/SNN-DNA-project/Prepocessing/sorted/1800_nM_AR_600'
    oneMinInFPS = 1200
    batch_size = 10
    num_workers = 0
    rnn_return_dict = {}
    featIn = oneMinInFPS
    epochs = 200
    modelSavePath = "./Models/RNN10Min/"
    totalRuntime = oneMinInFPS*10

    folders = [d[0] for d in os.walk("/users/kent/student/khood5/GitHub/SNN-DNA-project/Prepocessing/sorted")][1:]
    models = []
    for f in folders:
        print(f"training with {f.split('/')[-1]}")
        name = f.split('/')[-1]
        trainDataset, validDataset, testDataset = makeDataset(featIn, totalRuntime, f)
        model = RNNModel(featIn=featIn, capacity=int(300), hiddenLayers=4).to(device)
        MSE = nn.MSELoss(reduction = 'sum')
        adam = torch.optim.Adam(model.parameters(),lr=0.000001,weight_decay=1e-5)
        train(trainData=trainDataset, validData=validDataset, name=name, model=model, 
                lossfunction=MSE, optim=adam, return_dict=rnn_return_dict, epochs=epochs,
                device=device, savePath=modelSavePath)
        torch.save(model.state_dict(),f"{modelSavePath}{name}.pt")

        with open(f'{modelSavePath}rnn_valid_acc_{name}.json', 'w') as convert_file:
            convert_file.write(json.dumps(rnn_return_dict))
        print(rnn_return_dict)
        models.append((f"{modelSavePath}{name}.pt",testDataset))
    
    rnn_return_dict = {}
    for pair in models:
        model = pair[0]
        testDataset = pair[1]
        for moe in range(10,21):
            rnnmodel = RNNModel(featIn=featIn, capacity=int(300), hiddenLayers=4).to(device)
            name = f"{model.split('/')[-1]}_moe{moe}"
            print(name)
            test(testDataset, model, name, rnnmodel, rnn_return_dict, epochs=epochs, margin_of_error=moe, device=device)

    for model in rnn_return_dict.keys():
        rnn_return_dict[model]['targetPlot'] = rnn_return_dict[model]['targetPlot'].tolist()
        rnn_return_dict[model]['outputPlot'] = rnn_return_dict[model]['outputPlot'].tolist()
    
    print("\n\n\n\n")
    print("----")
    with open(f'{modelSavePath}rnn_test_acc.json', 'w') as convert_file:
        convert_file.write(json.dumps(rnn_return_dict))
    print("----")
        