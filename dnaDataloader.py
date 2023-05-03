import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
import random

class expermentDataloader(Dataset):
    def __init__(
        self,
        index_file: str, 
        data_path: str,
        startIndex = 0, # default to start from begining of list
        endIndex = None # default to grab the whole list (a.k.k arr[0:None] grabs whole thing)
    ):
        self.root_dir = data_path
        self.expermentSikeTrainsIndex = np.array(pd.read_csv(index_file,header=None))[startIndex:endIndex]
        self.spikeTrains = [
            f"{os.path.join(self.expermentSikeTrainsIndex[i][0])}" for i in range(len(self.expermentSikeTrainsIndex)) 
        ]
        self.targets = self.expermentSikeTrainsIndex[:, 1]

    def __getitem__(self, index):
        inputCSVlines = pd.read_csv(os.path.join(self.root_dir,self.spikeTrains[index]), header=None).to_numpy()
        targetCSVLines = self.targets[index]
        return inputCSVlines.flatten(), np.array([targetCSVLines])

    def __len__(self):
        return len(self.expermentSikeTrainsIndex)

# takes a expermentDataset and a rhs size splits the dataset into 2 set with rhs matching the desierd size and lhs having the remmaing elements 
def addData(lhs: list, rhs: list, expermentDataset: expermentDataloader, rhsSize=None):
    rhsSize = int(len(expermentDataset)*0.9) if rhsSize == None else rhsSize
    assert not rhsSize > len(expermentDataset)
    datasetIndexes = list(range(len(expermentDataset)))
    rhsIndexes = random.sample(datasetIndexes, k=rhsSize)
    for i in datasetIndexes:
        if i in rhsIndexes:
            rhs.append(expermentDataset[i])
        else:
            lhs.append(expermentDataset[i])