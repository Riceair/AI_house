from preprocessing.normalize import *
from preprocessing.symbolic import *
from config.data_setting import *
import torch

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets=None):
        self.data = np.array(data)
        self.is_train = False
        if targets is not None:
            self.targets = np.array(targets)
            self.is_train = True

    def __getitem__(self, index):
        datum = self.data[index]
        datum = np.array(datum, dtype=np.float32)
        datum = torch.from_numpy(datum)

        if self.is_train:
            target = self.targets[index]
            target = np.array(target, dtype=np.float32)
            target = torch.from_numpy(target)
            return datum, target
        else:
            return datum # not train mode

    def __len__(self):
        return len(self.data)