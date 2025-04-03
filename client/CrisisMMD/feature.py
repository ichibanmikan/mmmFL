import re
import os
import glob
import pandas as pd
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
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class CrisisMMDFeatureExtractor:
    def __init__(self, 
                 crisismmd_dir: str,
                 output_dir: str = "./crisismmd_features",  
                 text_model_name: str = "google/mobilebert-uncased",  
                 img_feature_dim: int = 1280,  
                 num_nodes: int = 30,  
                 test_ratio: float = 0.2  
                ):
        
        self.crisismmd_dir = Path(crisismmd_dir)
        self.output_dir = Path(output_dir)
        self.num_nodes = num_nodes
        self.test_ratio = test_ratio
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.mbn_v2 = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.mbn_v2.classifier = self.mbn_v2.classifier[:-1]
        self.mbn_v2 = self.mbn_v2.to(self.device)
        self.mbn_v2.eval() 

        self.img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])      
        
        self.tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
        self.mbm = MobileBertModel.from_pretrained("google/mobilebert-uncased")
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.mbm = self.mbm.to(self.device)
        self.mbm.eval()

        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        

        
        self.feature_len_dict = {
            'mobilenet_v2': 1280,
            'mobilebert': 512
        }

        self.img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
        self.label_dict = {
            'not_humanitarian':                         0, 
            'infrastructure_and_utility_damage':        1,
            'vehicle_damage':                           2, 
            'rescue_volunteering_or_donation_effort':   3,
            'other_relevant_information':               4, 
            'affected_individuals':                     5,
            'injured_or_dead_people':                   6, 
            'missing_or_found_people':                  7
        }
    def remove_url(self, text):
        text = re.sub(r'http\S+', '', text)
        # re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        return(text)
    # def _init_text_model(self, model_name):
    #     """"""
    #     self.text_tokenizer = MobileBertTokenizer.from_pretrained(model_name)
    #     self.text_model = MobileBertModel.from_pretrained(model_name).to(self.device)
    #     self.text_model.eval()
    
    # def _init_image_model(self):
    #     """"""
    #     self.img_model = models.mobilenet_v2(pretrained=True)
    #     self.img_model.classifier = self.img_model.classifier[:-1]  
    #     self.img_model = self.img_model.to(self.device)
    #     self.img_model.eval()
# def remove_url(text):
#     text = re.sub(r'http\S+', '', text)
#     # re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
#     return(text)
# train_text = remove_url(train_csv_data['tweet_text'].iloc[i]).strip() 

# def load_and_process_data(directory, task_prefix='task_informative'):
#     """
#     TSV
    
#     :
#         directory: TSV
#         task_prefix: ('task_informative')
    
#     :
#         train_df: DataFrame(traindev)
#         test_df: DataFrame
#     """
#     DataFrame
#     train_dfs = []
#     dev_dfs = []
#     test_dfs = []
    
#     
#     for filename in os.listdir(directory):
#         if filename.startswith(task_prefix) and filename.endswith('.tsv'):
#             filepath = os.path.join(directory, filename)
            
#             TSV
#             df = pd.read_csv(filepath, sep='\t')
            
#             
#             if 'train' in filename:
#                 train_dfs.append(df)
#             elif 'dev' in filename:
#                 dev_dfs.append(df)
#             elif 'test' in filename:
#                 test_dfs.append(df)
    
#     traindev
#     train_df = pd.concat(train_dfs + dev_dfs, ignore_index=True)
#     test_df = pd.concat(test_dfs, ignore_index=True)
    
#     
#     if 'tweet_text' in train_df.columns:
#         train_df['cleaned_text'] = train_df['tweet_text'].apply(remove_url)
#     if 'tweet_text' in test_df.columns:
#         test_df['cleaned_text'] = test_df['tweet_text'].apply(remove_url)
    
#     return train_df, test_df
    def _load_dataset(self):
        """"""
        train_csv_data = pd.read_csv(os.path.join(self.crisismmd_dir, "crisismmd_datasplit_all", 'task_humanitarian_text_img_train.tsv'), sep='\t')
        val_csv_data = pd.read_csv(os.path.join(self.crisismmd_dir, "crisismmd_datasplit_all", 'task_humanitarian_text_img_dev.tsv'), sep='\t')
        test_csv_data = pd.read_csv(os.path.join(self.crisismmd_dir, "crisismmd_datasplit_all", 'task_humanitarian_text_img_test.tsv'), sep='\t')
        

        train_data = []
        test_data = []

        combined_train_df = pd.concat([train_csv_data, val_csv_data], ignore_index=True)
        for i in range(len(combined_train_df)):
            train_data.append({
                'image_path': os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CrisisMMD_v2.0'), combined_train_df['image'].iloc[i]),
                'text': self.remove_url(combined_train_df['tweet_text'].iloc[i]).strip(),
                'label': combined_train_df['label'].iloc[i]
            })

        for i in range(len(test_csv_data)):
            test_data.append({
                'image_path': os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CrisisMMD_v2.0'), test_csv_data['image'].iloc[i]),
                'text': self.remove_url(test_csv_data['tweet_text'].iloc[i]).strip(),
                'label': test_csv_data['label'].iloc[i]
            })

        return train_data, test_data
        # data = []
        # for tsv in tsv_train_files:
        #     with open(tsv, 'r', encoding='utf-8') as f:
        #         
        #         next(f)  
        #         for line in f:
                    
        # train_text = remove_url(train_csv_data['tweet_text'].iloc[i]).strip()
        # # print(train_text)
        # train_data_dict[train_csv_data['image_id'].iloc[i]] = [
        #     train_csv_data['image_id'].iloc[i],
        #     str(Path(data_path).joinpath(train_csv_data['image'].iloc[i])),
        #     pm.label_dict[train_csv_data['label_image'].iloc[i]],
        #     train_text
        # ]
        #             fields = line.strip().split('\t')
        #             
        #             entry = {
        #                 'image_path': self.crisismmd_dir / fields[-1].strip(),
        #                 'text': fields[-3].strip(),
        #                 'label': fields[6].strip() if fields[6] else 'Not humanitarian',
        #             }
        #             
        #             if self._is_valid_sample(entry):
        #                 train_data.append(entry)
        # return train_data, test_data
    
    def _is_valid_sample(self, sample):
        """"""
        
        has_text = len(sample['text']) > 3  
        has_image = sample['image_path'].exists()
        return has_text or has_image
    
    def extract_text_feature(
        self, 
        input_str: str
    ) -> (np.array):
        """
        Extract features
        :param input_str: input string
        :return: return embeddings
        """
        with torch.no_grad():
            inputs = self.tokenizer(input_str, return_tensors="pt").to(self.device)
            outputs = self.mbm(**inputs)
            features = outputs.last_hidden_state.detach().cpu().numpy()[0]
        return features
    
    def extract_img_features(
        self, 
        img_path: str
    ) -> (np.array):
        """
        Extract the framewise feature from image
        :param img_path: image path
        :return: return features
        """
        input_image = Image.open(img_path).convert('RGB')
        input_tensor = self.img_transform(input_image)
        
        with torch.no_grad():
            input_data = input_tensor.to(self.device).unsqueeze(dim=0)
            features = self.mbn_v2(input_data).detach().cpu().numpy()
        return features

    def get_shape(self, lst):
        if isinstance(lst, list):
            return (len(lst),) + self.get_shape(lst[0]) if lst else (0,)
        else:
            return ()
    
    def process_dataset(self):
        """"""
        
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
                    'img': self.extract_img_features(sample['image_path']),
                    'label': self.label_dict[sample['label']]
                }
                train_data.append(elem_train)

            with open(self.output_dir / f"node_{node_id}.pkl", 'wb') as f:
                pickle.dump(train_data, f)

        test_data = []
       
        for sample in testset:
            elem_test = {
                'text': self.extract_text_feature(sample['text']),
                'img': self.extract_img_features(sample['image_path']),
                'label': self.label_dict[sample['label']]
            }
            test_data.append(elem_test)
        with open(self.output_dir / "test.pkl", 'wb') as f:
            pickle.dump(test_data, f)


if __name__ == "__main__":
    
    extractor = CrisisMMDFeatureExtractor(
        crisismmd_dir="",
        output_dir="",
        num_nodes=30,
        test_ratio=0.2
    )
    extractor.process_dataset()