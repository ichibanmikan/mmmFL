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
    def __init__(self, data_dir, modality):

        self.data_1 = np.load(os.path.join(data_dir, 'x1.npy'))
        self.data_2 = np.load(os.path.join(data_dir, 'x2.npy'))
        self.labels = np.load(os.path.join(data_dir, 'y.npy'))

class data_set(Dataset):
    def __init__(self, data_1, data_2, data_labels):
        super().__init__()
        self.data_1 = data_1
        self.data_2 = data_2
        self.labels = data_labels

    def __len__(self):
        return len(self.data_1)
    
    def __getitem__(self, index):  
        values = []
        
        if len(self.data_1) > 0:
            d_1 = self.data_1[index]
            d_1 = torch.unsqueeze(d_1, 0)
            values.append(d_1)
        if len(self.data_2) > 0:
            d_2 = self.data_2[index]
            d_2 = torch.unsqueeze(d_2, 0)
            values.append(d_2)

        return values, self.labels[index]

    
class data_factory:
    def __init__(self, data_dir, config, modality):
        self.rdata = rdata(data_dir, modality)
        self.config = config
    def get_dataset(self):
        board_0 = round(len(self.rdata.data_1) * 0.9)
        
        train_data_1 = torch.tensor(self.rdata.data_1[:board_0], dtype=torch.float32)
        train_data_2 = torch.tensor(self.rdata.data_2[:board_0], dtype=torch.float32)
        train_labels = torch.tensor(self.rdata.labels[:board_0], dtype=torch.long)
         
        # test_data_1 = torch.tensor(self.rdata.data_1[board_0 : board_1], dtype=torch.float32)
        # test_data_2 = torch.tensor(self.rdata.data_2[board_0 : board_1], dtype=torch.float32)
        
        # test_labels = torch.tensor(self.rdata.labels[board_0 : board_1], dtype=torch.long)  
              
        valid_data_1 = torch.tensor(self.rdata.data_1[board_0:], dtype=torch.float32)
        valid_data_2 = torch.tensor(self.rdata.data_2[board_0:], dtype=torch.float32)
        
        valid_labels = torch.tensor(self.rdata.labels[board_0:], dtype=torch.long)
        # print('aaa ', len(self.rdata.data_1))
        datasets = [ data_set(train_data_1, train_data_2, train_labels), 
                     data_set(valid_data_1, valid_data_2, valid_labels)
                ]
        dataloaders = [ DataLoader(datasets[0], shuffle=True, drop_last=True, batch_size=self.config.batch_size, num_workers=self.config.num_workers), 
                       DataLoader(datasets[1], drop_last=True, batch_size=self.config.batch_size, num_workers=self.config.num_workers)]
        # return datasets, dataloaders
        return dataloaders