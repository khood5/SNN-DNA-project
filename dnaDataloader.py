import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# import slayer from lava-dl
import lava.lib.dl.slayer as slayer

class expermentDataloader(Dataset):
    def __init__(
        self,
        index_file: str, 
        data_path: str,
    ):
        super(expermentDataloader, self).__init__()
        self.root_dir = data_path
        self.expermentSikeTrainsIndex = pd.read_csv(index_file) # self.landmarks_frame = pd.read_csv(csv_file)
        self.inputs = [
            f"{os.path.join(self.expermentSikeTrainsIndex.iloc[i, 0])}" for i in range(len(self.expermentSikeTrainsIndex)) 
        ]
        self.targets = [
            f"{os.path.join(self.expermentSikeTrainsIndex.iloc[i, 1])}" for i in range(len(self.expermentSikeTrainsIndex)) 
        ]

    def _fileToSlayerEvents(self, fileName: str):
        CSVlines = pd.read_csv(os.path.join(self.root_dir,fileName)).to_numpy()
        events = np.array(torch.FloatTensor(CSVlines))
        
        x_event = np.flip(events[:, 0])
        y_event = None
        c_event = torch.zeros(len(x_event), )
        t_event = np.flip(events[:, 1])
        return slayer.io.Event(x_event,y_event,c_event,t_event)

    def __getitem__(self, index):
        input = self._fileToSlayerEvents(self.inputs[index])
        target = self._fileToSlayerEvents(self.targets[index])
        
        return (
            input.fill_tensor(torch.zeros(1, 1, 200, 25000)).squeeze(), # input spike train
            target.fill_tensor(torch.zeros(1, 1, 200, 25000)).squeeze() # target spike train
        )
        # return torch.FloatTensor(CSVlines.flatten()), int(eventClass)

    def getSlayerEvents(self, index: int):
        input = self._fileToSlayerEvents(self.inputs[index])
        target = self._fileToSlayerEvents(self.targets[index])
        
        return (
            input, # input spike train
            target # target spike train
        )

    def __len__(self):
        return len(self.expermentSikeTrainsIndex)
