import numpy as np
from tonic import Dataset, transforms
import pandas as pd
import os

sensor_size = (
    200,
    68,
    2,
)  # the sensor size of the event camera or the number of channels of the silicon cochlear that was used
ordering = (
    "txyp"  # the order in which your event channels are provided in your recordings
)
dtype=np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])


class gridSTAGEDataset(Dataset):
    def __init__(
        self,
        index_file: str, 
        data_path: str,
        train=True,
        transform=None,
        target_transform=None,
    ):
        super(gridSTAGEDataset, self).__init__(
            save_to='./', transform=transform, target_transform=target_transform
        )
        self.train = train

        # replace the strings with your training/testing file locations or pass as an argument
        # if train:
        self.scenarios = pd.read_csv(index_file) # self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = data_path
        self.transform = transform
        self.filenames = [
            f"{os.path.join(self.root_dir, self.scenarios.iloc[i, 0])}/events.csv" for i in range(len(self.scenarios))  # type: ignore // suppress warning 
        ]
        self.scenarios_types = self.scenarios.iloc[:, 1]
        # else:
        #     raise NotImplementedError

    def __getitem__(self, index):
        CSVlines = pd.read_csv(self.filenames[index]).to_numpy()
        eventClass = self.scenarios_types[index]
        events = np.zeros(len(CSVlines), dtype=dtype)
        # events = np.load(self.filenames[index])
        events["x"] = CSVlines[ :,1]
        events["y"] = CSVlines[ :,2]
        events["p"] = CSVlines[ :,3]
        events["t"] = CSVlines[ :,0]
        if self.transform is not None:
            events = self.transform(events)

        return events, eventClass

    def __len__(self):
        return len(self.filenames)
