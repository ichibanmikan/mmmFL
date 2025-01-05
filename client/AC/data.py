import os
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import torch

# 获取当前进程的内存使用情况
# def get_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info()
#     return mem.rss  # 返回常驻内存集的大小（以字节为单位）
# def get_total_size(obj):
#     if isinstance(obj, (list, tuple)):
#         return sys.getsizeof(obj) + sum(get_total_size(i) for i in obj)
#     return sys.getsizeof(obj)

'''modality是一个列表，记录了需要读取的所有的模态数据'''

class data_set(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.data_label = torch.tensor(np.load(os.path.join(data_dir, './label.npy')), dtype=torch.long)
    def __len__(self):
        return len(self.data_label)
    
    def __getitem__(self, index):  
        data_1 = np.load(os.path.join(self.data_dir, './audio/' + str(index) + '.npy'))
        data_2 = np.load(os.path.join(self.data_dir, './depth/' + str(index) + '.npy'))
        data_3 = np.load(os.path.join(self.data_dir, './radar/' + str(index) + '.npy'))

        data_1 = torch.tensor(data_1, dtype=torch.float16)
        data_2 = torch.tensor(data_2, dtype=torch.float16)
        data_3 = torch.tensor(data_3, dtype=torch.float16)
        
        # data_1 = torch.unsqueeze(data_1, 0)
        data_2 = torch.unsqueeze(data_2, 0)
        # data_3 = torch.unsqueeze(data_3, 0)
        
        return [data_1, data_2, data_3], self.data_label[index]

    
class get_dataloaders:
    def __init__(self, data_dir, config):
        self.data_dir = data_dir
        self.config = config
        
    def get_dataloaders(self):
        ds = data_set(self.data_dir)
        train_size = int(0.9 * len(ds))
        val_size = len(ds) - train_size

        train_dataset, val_dataset = random_split(ds, [train_size, val_size])
        dls = [DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=self.config.batch_size, num_workers=self.config.num_workers), 
               DataLoader(val_dataset, shuffle=True, drop_last=True, batch_size=self.config.batch_size, num_workers=self.config.num_workers)]
        return dls