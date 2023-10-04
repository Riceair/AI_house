from preprocessing.normalize import *
from preprocessing.symbolic import *
from config.data_setting import *
import torch

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = np.array(data)
        self.targets = np.array(targets)

    def __getitem__(self, index):
        datum = self.data[index]
        target = self.targets[index]
        datum = torch.from_numpy(datum)
        target = torch.from_numpy(target)
        return datum, target

    def __len__(self):
        return len(self.datum)