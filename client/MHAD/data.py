import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import psutil

# 获取当前进程的内存使用情况
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    return mem.rss  # 返回常驻内存集的大小（以字节为单位）
# def get_total_size(obj):
#     if isinstance(obj, (list, tuple)):
#         return sys.getsizeof(obj) + sum(get_total_size(i) for i in obj)
#     return sys.getsizeof(obj)

'''modality是一个列表，记录了需要读取的所有的模态数据'''
class rdata:
    def __init__(self, data_dir):
        self.data_1 = torch.tensor(np.load(os.path.join(data_dir, 'x1.npy')).astype(np.float32))
        self.data_2 = torch.tensor(np.load(os.path.join(data_dir, 'x2.npy')).astype(np.float32))
        self.labels = torch.tensor(np.load(os.path.join(data_dir, 'y.npy')).astype(np.int64))

class data_set(Dataset):
    def __init__(self, data_1, data_2, labels, noise_std = 0.01):
        super().__init__()
        self.data_1 = data_1
        self.data_2 = data_2
        self.labels = labels
        
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
        self.rd = rdata(data_dir)
        self.config = config

    def get_dataset(self):
        board_0 = round(len(self.rd.data_1) * 0.9)
        indices = np.random.choice(len(self.rd.data_1), size=board_0, replace=False)
        train_ds = data_set(self.rd.data_1[indices], self.rd.data_2[indices], self.rd.labels[indices])
        valid_ds = data_set(self.rd.data_1[~indices], self.rd.data_2[~indices], self.rd.labels[~indices])

        dataloaders = [ DataLoader(train_ds, shuffle=True, batch_size=self.config.batch_size, num_workers=self.config.num_workers), 
                       DataLoader(valid_ds, batch_size=self.config.batch_size, num_workers=self.config.num_workers)]
        return dataloaders