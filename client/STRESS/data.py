from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import numpy as np

class data_set(Dataset):
    def __init__(self, data_csv):
        super().__init__()
        self.data = pd.read_csv(data_csv).to_numpy()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index][0:6], dtype=torch.float32), \
            torch.tensor(self.data[index][6], dtype=torch.long)

    
class data_factory:
    def __init__(self, data_csv, config):
        self.data_set = data_set(data_csv)
        self.config = config
        
    def get_dataset(self):
        # return datasets, dataloaders
        return  DataLoader(self.data_set, shuffle=True, drop_last=True, \
            batch_size=self.config.batch_size, num_workers=self.config.num_workers)
