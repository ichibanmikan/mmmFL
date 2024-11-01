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
    def __init__(self, data_dir, state, modality):
        data_list_1 = []
        data_list_2 = []
        data_list_3 = []
        labels_list = []
        
        # for root, d, files in os.walk(data_dir):
        if state == 1 or len(modality) == 1:
            x_tr = np.load(os.path.join(data_dir, modality[0] + '.npz'))
            y_tr = np.load(os.path.join(data_dir, 'rf.npz'))
            data_list_1 = (x_tr[modality[0]])
            labels_list = (y_tr['rf'])
        elif state == 2:
            if len(modality) == 2:
                if modality[0]=='gps' or (modality[0]=='lidar' and modality[1]=='image'):
                    x_tr_1 = np.load(os.path.join(data_dir, modality[0] + '.npz'))
                    x_tr_2 = np.load(os.path.join(data_dir, modality[1] + '.npz'))
                else:
                    x_tr_2 = np.load(os.path.join(data_dir, modality[0] + '.npz'))
                    x_tr_1 = np.load(os.path.join(data_dir, modality[1] + '.npz'))
                y_tr = np.load(os.path.join(data_dir, 'rf.npz'))
                data_list_1 = (x_tr_1[modality[0]])
                data_list_2 = (x_tr_2[modality[1]])
                labels_list = (y_tr['rf'])
            else:
                x_tr_1 = np.load(os.path.join(data_dir, 'gps.npz'))
                x_tr_2 = np.load(os.path.join(data_dir, 'lidar.npz'))
                x_tr_3 = np.load(os.path.join(data_dir, 'image.npz'))
                y_tr = np.load(os.path.join(data_dir, 'rf.npz'))
                # for i in range(len(x_tr_1['gps'])):
                data_list_1 = (x_tr_1['gps'])
                data_list_2 = (x_tr_2['lidar'])
                data_list_3 = (x_tr_3['image'])
                labels_list = (y_tr['rf'])
                    
        self.labels = np.array(labels_list)
        if len(data_list_1) > 0:
            self.data_1 = np.array(data_list_1, dtype='float')
        else:
            self.data_1 = np.empty((0,))  # 或者选择其他合适的默认值
            
        if len(data_list_2) > 0:
            self.data_2 = np.array(data_list_2, dtype='float')
        else:
            self.data_2 = np.empty((0,))  # 或者选择其他合适的默认值

        if len(data_list_3) > 0:
            self.data_3 = np.array(data_list_3, dtype='float')
        else:
            self.data_3 = np.empty((0,))  # 或者选择其他合适的默认值

    # def count_files_in_directory(self, dir_path):
    #     return sum(1 for item in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, item)))
    def has_npz_file(self, d):
        for entry in os.listdir(d):
            if os.path.isfile(os.path.join(d, entry)) and entry.endswith('.npz'):
                return True
        return False

class data_set(Dataset):
    def __init__(self, data_1, data_2, data_3, data_labels):
        super().__init__()
        self.data_1 = data_1
        self.data_2 = data_2
        self.data_3 = data_3
        self.labels = data_labels

    def __len__(self):
        return len(self.data_1)
    
    def __getitem__(self, index):  
        values = []
        
        if len(self.data_1) > 0:
            values.append(self.data_1[index])
        if len(self.data_2) > 0:
            values.append(self.data_2[index])
        if len(self.data_3) > 0:
            values.append(self.data_3[index])

        return values, self.labels[index]

    
class data_factory:
    def __init__(self, data_dir, config, state, modality):
        self.rdata = rdata(data_dir, state, modality)
        self.config = config
    def get_dataset(self):
        board_0 = round(len(self.rdata.data_1) * 0.8)
        board_1 = round(len(self.rdata.data_1) * 0.8)+round(len(self.rdata.data_1) * 0.15)
        
        train_data_1 = torch.tensor(self.rdata.data_1[:board_0])
        train_data_2 = torch.tensor(self.rdata.data_2[:board_0])
        train_data_3 = torch.tensor(self.rdata.data_3[:board_0])  
        train_labels = torch.tensor(self.rdata.labels[:board_0])
         
        test_data_1 = torch.tensor(self.rdata.data_1[board_0 : board_1])
        test_data_2 = torch.tensor(self.rdata.data_2[board_0 : board_1])
        test_data_3 = torch.tensor(self.rdata.data_3[board_0 : board_1])
        
        test_labels = torch.tensor(self.rdata.labels[board_0 : board_1])  
              
        valid_data_1 = torch.tensor(self.rdata.data_1[board_1:])
        valid_data_2 = torch.tensor(self.rdata.data_2[board_1:])
        valid_data_3 = torch.tensor(self.rdata.data_3[board_1:])
        
        valid_labels = torch.tensor(self.rdata.labels[board_1:])
        # print('aaa ', len(self.rdata.data_1))
        datasets = [data_set(train_data_1, train_data_2, train_data_3, train_labels), data_set(test_data_1, test_data_2, test_data_3, test_labels), data_set(valid_data_1, valid_data_2, valid_data_3, valid_labels)]
        dataloaders = [DataLoader(datasets[0], shuffle=True, batch_size=self.config.batch_size), DataLoader(datasets[1], shuffle=True, batch_size=self.config.batch_size), DataLoader(datasets[2], batch_size=self.config.batch_size)]
        # return datasets, dataloaders
        return dataloaders