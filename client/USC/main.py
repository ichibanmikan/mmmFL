# Copyright 2024 ichibanmikan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from USC.data import *
from USC.worker import *

class Config:
    def __init__(self, config_path) -> None:
        self.config_path = config_path
        self.load_config()

    def load_config(self) -> None:
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)

        self.print_freq = config_data.get('print_freq', 5)
        self.save_freq = config_data.get('save_freq', 20)
        self.batch_size = config_data.get('batch_size', 16)
        self.num_workers = config_data.get('num_workers', 16)
        self.epochs = config_data.get('epochs', 99)
        self.learning_rate = config_data.get('learning_rate', 0.001)
        self.lr_decay_rate = config_data.get('lr_decay_rate', 0.9)
        self.weight_decay = config_data.get('weight_decay', 0.0001)
        self.momentum = config_data.get('momentum', 0.9)
        self.num_classes = config_data.get('num_classes', 13)
        self.total_epochs = config_data.get('total_epochs', 200)

    def __repr__(self) -> str:
        return f"Config({self.__dict__})"

class USC_main:
    def __init__(self, modality, node_id):
        self.modality = modality
        self.now_loss = 999
        self.config = Config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'))
        
        self.model = MMModel(1, self.config.num_classes)      

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.model = self.model.to(device)
        data_f = data_factory(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/node_'+f"{node_id}/"), self.config, self.modality)
        train_loader, valid_loader = data_f.get_dataset()
        self.tr = Trainer(self.config, self.model, train_loader, valid_loader, device)
        
    def main(self):

        self.now_loss = self.tr.train()
        print(self.tr.best_acc)
        
        return self.get_model_param()
        
    def get_model_param(self):
    
        params = []
        for param in self.model.parameters():
            if torch.cuda.is_available() or torch.backends.mps.is_available():
                params.extend(param.view(-1).cpu().detach().numpy())
            else:
                params.extend(param.view(-1).detach().numpy())
            # print(param)

        # model_params = params.cpu().numpy()
        model_params = np.array(params)
        print("Shape of model weight: ", model_params.shape)#39456

        return model_params

    def reset_model_parameter(self, new_params):
        
        temp_index = 0

        with torch.no_grad():
            for param in self.model.parameters():

                # print(param.shape)

                if len(param.shape) == 2:

                    para_len = int(param.shape[0] * param.shape[1])
                    # print(para_len)
                    # print(temp_index)
                    # print(len(new_params))
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    # print(len(temp_weight))
                    param.copy_(torch.Tensor(temp_weight.reshape(param.shape[0], param.shape[1])))
                    temp_index += para_len

                elif len(param.shape) == 4:

                    para_len = int(param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3])
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2], param.shape[3])))
                    temp_index += para_len  

                elif len(param.shape) == 5:

                    para_len = int(param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3] * param.shape[4])
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2], param.shape[3], param.shape[4])))
                    temp_index += para_len  

                else:

                    para_len = param.shape[0]
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight))
                    temp_index += para_len      

    def sample_time(self):
        return self.tr.sample_one_epoch()
    
    def save_model(self, round):
        str = ""
        for i in range(len(self.modality)):
            str += self.modality[i]
        
        self.tr.train_tools.save_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/', str + '.pth'))
        