import re
import os
import torch.nn as nn
import json
import pickle
import torch
import numpy as np
from PIL import Image
import random
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from transformers import MobileBertTokenizer, MobileBertModel
from torchvision import models
from torchvision.models import densenet161

from transformers import DebertaTokenizer, DebertaModel
import torch


class CrisisMMDFeatureExtractor:
    def __init__(self, 
                 hm_dir: str,
                 output_dir: str = "./crisismmd_features",
                 text_model_name: str = "google/mobilebert-uncased",
                 img_feature_dim: int = 1280,
                 num_nodes: int = 30,
                ):
        
        self.hm_dir = Path(hm_dir)
        self.output_dir = Path(output_dir)
        self.num_nodes = num_nodes
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dsnt = densenet161(weights = "IMAGENET1K_V1")
        self.dsnt.classifier = nn.Identity()
        self.dsnt = self.dsnt.to(self.device)
        self.dsnt.eval() 

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])      
        
        self.tokenizer = DebertaTokenizer.from_pretrained('')
        self.mbm = DebertaModel.from_pretrained('')
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.mbm = self.mbm.to(self.device)
        self.mbm.eval()

        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        

        
        self.feature_len_dict = {
            'mobilenet_v2': 1280,
            'mobilebert': 512
        }


    def _load_dataset(self):
        train_data = []
        with open(os.path.join(self.hm_dir, 'train.jsonl'), 'r') as f:
            for line in f:
                train_data.append(json.loads(line))

        test_data = []
        with open(os.path.join(self.hm_dir, 'test.jsonl'), 'r') as f:
            for line in f:
                test_data.append(json.loads(line))

        return train_data, test_data
    
    def extract_text_feature(
        self, 
        input_str: str
    ) -> (np.array):

        with torch.no_grad():
            inputs = self.tokenizer(input_str, return_tensors="pt").to(self.device)
            outputs = self.mbm(**inputs)
            features = outputs.last_hidden_state.detach().cpu().numpy()[0]
        return features
    
    def extract_img_features(
        self, 
        img_path: str
    ) -> (np.array):

        input_image = Image.open(img_path).convert('RGB')
        input_tensor = self.img_transform(input_image)
        
        with torch.no_grad():
            input_data = input_tensor.to(self.device).unsqueeze(dim=0)
            features = self.dsnt(input_data).detach().cpu().numpy()
        return features

    def get_shape(self, lst):
        if isinstance(lst, list):
            return (len(lst),) + self.get_shape(lst[0]) if lst else (0,)
        else:
            return ()
    
    def process_dataset(self):
        trainset, testset = self._load_dataset()

        node_size = len(trainset) // self.num_nodes
        for node_id in tqdm(range(self.num_nodes)):
            train_data = []
            
            
            start = node_id * node_size
            end = (node_id + 1) * node_size
            node_data = trainset[start:end]

            for sample in node_data:
                elem_train = {
                    'text': self.extract_text_feature(sample['text']),
                    'img': self.extract_img_features(os.path.join(self.hm_dir, sample['img'])),
                    'label': sample['label']
                }
                train_data.append(elem_train)

            with open(self.output_dir / f"node_{node_id}.pkl", 'wb') as f:
                pickle.dump(train_data, f)

        test_data = []
       
        for sample in testset:
            elem_test = {
                'text': self.extract_text_feature(sample['text']),
                'img': self.extract_img_features(os.path.join(self.hm_dir, sample['img'])),
                'label': sample['label']
            }
            test_data.append(elem_test)
        with open(self.output_dir / "test.pkl", 'wb') as f:
            pickle.dump(test_data, f)


if __name__ == "__main__":
    
    extractor = CrisisMMDFeatureExtractor(
        hm_dir="",
        output_dir="",
        num_nodes=30,
    )
    extractor.process_dataset()