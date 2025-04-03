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

'''modality，'''

# class rdata:
#     def __init__(self, data_dir, modality):
#         data_list_1 = []
#         data_list_2 = []
#         data_list_3 = []
#         labels_list = []
        
#         # for root, d, files in os.walk(data_dir):
#         if len(modality) == 1:
#             x_tr = np.load(os.path.join(data_dir, modality[0] + '.npz'))
#             y_tr = np.load(os.path.join(data_dir, 'rf.npz'))
#             data_list_1 = (x_tr[modality[0]])
#             labels_list = (y_tr['rf'])
#         else:
#             if len(modality) == 2:
#                 if modality[0]=='gps' or (modality[0]=='lidar' and modality[1]=='image'):
#                     x_tr_1 = np.load(os.path.join(data_dir, modality[0] + '.npz'))
#                     x_tr_2 = np.load(os.path.join(data_dir, modality[1] + '.npz'))
#                 else:
#                     x_tr_2 = np.load(os.path.join(data_dir, modality[0] + '.npz'))
#                     x_tr_1 = np.load(os.path.join(data_dir, modality[1] + '.npz'))
#                 y_tr = np.load(os.path.join(data_dir, 'rf.npz'))
#                 data_list_1 = (x_tr_1[modality[0]])
#                 data_list_2 = (x_tr_2[modality[1]])
#                 labels_list = (y_tr['rf'])
#             else:
#                 x_tr_1 = np.load(os.path.join(data_dir, 'gps.npz'))
#                 x_tr_2 = np.load(os.path.join(data_dir, 'lidar.npz'))
#                 x_tr_3 = np.load(os.path.join(data_dir, 'image.npz'))
#                 y_tr = np.load(os.path.join(data_dir, 'rf.npz'))
#                 # for i in range(len(x_tr_1['gps'])):
#                 data_list_1 = (x_tr_1['gps'])
#                 data_list_2 = (x_tr_2['lidar'])
#                 data_list_3 = (x_tr_3['image'])
#                 labels_list = (y_tr['rf'])
                    
#         self.labels = np.array(labels_list)
#         if len(data_list_1) > 0:
#             self.data_1 = np.array(data_list_1, dtype='float')
#         else:
#             self.data_1 = np.empty((0,))  # 
            
#         if len(data_list_2) > 0:
#             self.data_2 = np.array(data_list_2, dtype='float')
#         else:
#             self.data_2 = np.empty((0,))  # 

#         if len(data_list_3) > 0:
#             self.data_3 = np.array(data_list_3, dtype='float')
#         else:
#             self.data_3 = np.empty((0,))  # 

#     # def count_files_in_directory(self, dir_path):
#     #     return sum(1 for item in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, item)))
#     def has_npz_file(self, d):
#         for entry in os.listdir(d):
#             if os.path.isfile(os.path.join(d, entry)) and entry.endswith('.npz'):
#                 return True
#         return False

class data_set(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_1 = np.load(os.path.join(data_dir, 'gps.npz'))['gps'].astype(np.float32)
        self.data_2 = np.load(os.path.join(data_dir, 'lidar.npz'))['lidar'].astype(np.float32)
        self.data_3 = np.load(os.path.join(data_dir, 'image.npz'))['image'].astype(np.float32)
        self.labels = np.load(os.path.join(data_dir, 'rf.npz'))['rf'].astype(np.int64)

    def __len__(self):
        return len(self.data_1)
    
    def __getitem__(self, index):  

        return self.data_1[index], self.data_2[index], self.data_3[index], self.labels[index]

    
class data_factory:
    def __init__(self, data_dir, config):
        self.ds = data_set(data_dir)
        self.config = config
    def get_dataset(self):
        return DataLoader(self.ds, shuffle=True, drop_last=True, \
            batch_size=self.config.batch_size, num_workers=self.config.num_workers)
                       