import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import psutil

# 
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    return mem.rss  # （）
# def get_total_size(obj):
#     if isinstance(obj, (list, tuple)):
#         return sys.getsizeof(obj) + sum(get_total_size(i) for i in obj)
#     return sys.getsizeof(obj)

class data_set(Dataset):
    def __init__(self, data_dir, noise_std = 0.01):
        super().__init__()
        self.data_1 = torch.tensor(np.load(os.path.join(data_dir, 'x1.npy')).astype(np.float32))
        self.data_2 = torch.tensor(np.load(os.path.join(data_dir, 'x2.npy')).astype(np.float32))
        self.labels = torch.tensor(np.load(os.path.join(data_dir, 'y.npy')).astype(np.int64))
        
        self.noise_std = noise_std

    def __len__(self):
        return len(self.data_1)
    
    def __getitem__(self, index): 
        d1 = self.data_1[index]
        d2 = self.data_2[index]
        d1 = d1 + torch.randn_like(d1) * self.noise_std
        d2 = d2 + torch.randn_like(d2) * self.noise_std
        return d1.unsqueeze(0), d2.unsqueeze(0), self.labels[index]

    
class data_factory:
    def __init__(self, data_dir, config):
        self.ds = data_set(data_dir)
        self.config = config
        self.sample_length = len(self.ds)

    def get_dataset(self):
        dataloaders = DataLoader(self.ds, shuffle=True, batch_size=self.config.batch_size, \
            num_workers=self.config.num_workers)
        return dataloaders