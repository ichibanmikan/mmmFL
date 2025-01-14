import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from sklearn.model_selection import train_test_split

class data_set(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_1 = torch.tensor(np.load(os.path.join(data_dir, 'acc.npy')), dtype=torch.float32)
        self.data_2 = torch.tensor(np.load(os.path.join(data_dir, 'gyr.npy')), dtype=torch.float32)
        self.labels = torch.tensor(np.load(os.path.join(data_dir, 'action.npy')), dtype=torch.long)

    def augment_data(self, data):
        noise = torch.randn_like(data) * 0.01
        data = data + noise

        if torch.rand(1) > 0.5:
            shift = torch.randint(-10, 10, (1,))
            data = torch.roll(data, shifts=shift.item(), dims=-1)
            
        return data

    def __len__(self):
        return len(self.data_1)
    
    def __getitem__(self, index):
        acc = self.data_1[index]
        gyr = self.data_2[index]

        acc = self.augment_data(acc)
        gyr = self.augment_data(gyr)
            
        return acc.unsqueeze(0), gyr.unsqueeze(0), self.labels[index]

    
class data_factory:
    def __init__(self, data_dir, config):
        self.data_set = data_set(data_dir)
        self.config = config
        
    def get_dataset(self):
        # return datasets, dataloaders
        return  DataLoader(self.data_set, shuffle=True, drop_last=True, \
            batch_size=self.config.batch_size,num_workers=self.config.num_workers)
