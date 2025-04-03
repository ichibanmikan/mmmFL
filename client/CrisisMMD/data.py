import pickle
from torch.utils.data import DataLoader, Dataset
import torch
import os
import numpy as np

# class DataSet(Dataset):
#     def __init__(self, data_pkl, device = "cuda"):
#         self.data = pickle.load(data_pkl)
#         self.device = device

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         text_feature = self.data[index]["text"]
#         img_feature = self.data[index]["img"]
#         label = self.data[index]["label"]
#         return torch.tensor(text_feature, dtype=torch.float32), \
#                 torch.tensor(img_feature, dtype=torch.float32), \
#                     torch.tensor(label, dtype=torch.long)

def pad_tensor(vec, pad, dim=0):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size, dtype=vec.dtype, device=vec.device)], dim=dim)

def collate_fn_padd(batch):
    texts, imgs, text_len, img_len, labels = zip(*batch)

    max_text_len = max(t.shape[0] for t in texts)
    max_img_len = max(i.shape[0] for i in imgs) if imgs[0].ndim > 0 else 0

    padded_texts = []
    for text in texts:
        if text.ndim == 1:
            padded_text = pad_tensor(text, max_text_len, 0)
        else:
            padded_text = pad_tensor(text, max_text_len, 0)
        padded_texts.append(padded_text)

    padded_imgs = []
    for img in imgs:
        if img.ndim == 1:
            padded_img = pad_tensor(img, max_img_len, 0) if max_img_len > 0 else img
        else:
            padded_img = pad_tensor(img, max_img_len, 0) if max_img_len > 0 else img
        padded_imgs.append(padded_img)

    padded_texts = torch.stack(padded_texts, dim=0)
    padded_imgs = torch.stack(padded_imgs, dim=0) if max_img_len > 0 else torch.stack(imgs, dim=0)
    labels = torch.stack(labels, dim=0)
    
    text_len = torch.stack(text_len, dim=0)
    img_len = torch.stack(img_len, dim=0)
    
    return padded_texts, padded_imgs, text_len, img_len, labels

class DataSet(Dataset):
    def __init__(self, data_dir, device="cuda"):
        self.data = []
        for i in range(30):
            pkl_path = os.path.join(data_dir, 'node_' + f"{i}.pkl")
            if os.path.exists(pkl_path):
                self.data.extend(pickle.load(open(pkl_path, "rb")))
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text_feature = self.data[index]["text"]
        img_feature = self.data[index]["img"]
        label = self.data[index]["label"]
        return (torch.tensor(text_feature, dtype=torch.float32),
                torch.tensor(img_feature, dtype=torch.float32),
                torch.tensor(text_feature.shape[0], dtype=torch.long),
                torch.tensor(img_feature.shape[0], dtype=torch.long),
                torch.tensor(label, dtype=torch.long))

class DataFactory:
    def __init__(self, data_pkl, config):
        self.dataset = DataSet(data_pkl)
        self.config = config
    def get_dataloader(self):
        train_loader = DataLoader(
            self.dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            collate_fn=collate_fn_padd
        )
        return train_loader