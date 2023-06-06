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
        length: None
    ):
        self.root_dir = data_path
        self.expermentMovieFrames = np.array(pd.read_csv(index_file,header=None))
        self.frames = [
            f"{os.path.join(self.expermentMovieFrames[i][0])}" for i in range(len(self.expermentMovieFrames)) 
        ]
        self.targets = self.expermentMovieFrames[:, 1]
        self.length = length

    def __getitem__(self, index):
        expermentMovieFrames = pd.read_csv(os.path.join(self.root_dir,self.frames[index]), header=None).to_numpy()
        expermentMovieFrames = expermentMovieFrames[:self.length]
        totalNumberOfEvents = self.targets[index]
        return expermentMovieFrames.flatten(), np.array([totalNumberOfEvents])

    def __len__(self):
        return len(self.expermentMovieFrames)

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