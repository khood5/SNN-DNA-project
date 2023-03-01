import numpy as np
import pandas as pd
import os
from copy import deepcopy

class expermentDataloader:
    def __init__(
        self,
        index_file: str, 
        data_path: str,
    ):
        self.root_dir = data_path
        self.expermentSikeTrainsIndex = pd.read_csv(index_file) # self.landmarks_frame = pd.read_csv(csv_file)
        self.spikeTrains = [
            f"{os.path.join(self.root_dir, self.expermentSikeTrainsIndex.iloc[i, 0])}" for i in range(len(self.expermentSikeTrainsIndex)) 
        ]
        self.expermentClasses = self.expermentSikeTrainsIndex.iloc[:, 1]

    def __getitem__(self, index):
        CSVlines = pd.read_csv(self.spikeTrains[index]).to_numpy()
        eventClass = self.expermentClasses[index]
        events = np.zeros(len(CSVlines), dtype=int)
        
        return deepcopy((events, eventClass))

    def __len__(self):
        return deepcopy(len(self.expermentSikeTrainsIndex))
