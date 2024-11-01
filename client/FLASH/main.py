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
from FLASH.data import *
from FLASH.worker import *

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
        self.lr_decay_epochs = config_data.get('lr_decay_epochs', '50,100,150')
        self.lr_decay_rate = config_data.get('lr_decay_rate', 0.9)
        self.weight_decay = config_data.get('weight_decay', 0.0001)
        self.momentum = config_data.get('momentum', 0.9)
        self.cosine = config_data.get('cosine', True)
        self.num_classes = config_data.get('num_classes', 12)

    def __repr__(self) -> str:
        return f"Config({self.__dict__})"

class FLASH_main:
    
    def __init__(self, state, modality):
        self.state = state
        self.modality = modality
        self.config = Config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'))
        if self.state == 1:
            self.model = MySingleModel(self.config.num_classes, self.modality[0])
        else:
            if len(self.modality) == 1:
                self.model = MySingleModel(self.config.num_classes, self.modality[0])
            elif len(self.modality) == 2:
                self.model = My2Model(self.config.num_classes, self.modality[0] + ' ' + self.modality[1])
            else:
                self.model = My3Model(self.config.num_classes)      
        
    def main(self, node_id):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.model = self.model.to(device)
        data_f = data_factory(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/node_'+f"{node_id}/"), self.config, self.state, self.modality)
        train_loader, valid_loader, test_loader = data_f.get_dataset()

        self.tr = Trainer(self.config, self.model, train_loader, valid_loader, device, self.state)

        self.tr.train()
        print(self.tr.best_acc)
        
        return self.get_model_update()
        
    def get_model_update(self):
        
        params = []
        for param in self.model.parameters():
            if torch.cuda.is_available() or torch.backends.mps.is_available():
                params.extend(param.view(-1).cpu().detach().numpy())
            else :
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
                if len(param.shape) == 2:

                    para_len = int(param.shape[0] * param.shape[1])
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight.reshape(param.shape[0], param.shape[1])).to(param.device))
                    temp_index += para_len

                elif len(param.shape) == 3:

                    para_len = int(param.shape[0] * param.shape[1] * param.shape[2])
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2])).to(param.device))
                    temp_index += para_len 

                elif len(param.shape) == 4:

                    para_len = int(param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3])
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2], param.shape[3])).to(param.device))
                    temp_index += para_len  

                elif len(param.shape) == 5:

                    para_len = int(param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3] * param.shape[4])
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2], param.shape[3], param.shape[4])).to(param.device))
                    temp_index += para_len  

                else:

                    para_len = param.shape[0]
                    temp_weight = new_params[temp_index : temp_index + para_len].astype(float)
                    param.copy_(torch.Tensor(temp_weight).to(param.device))
                    temp_index += para_len      
    
    def save_model(self, round):
        str = ""
        for i in range(len(self.modality)):
            str += self.modality[i]
        
        if(self.state == 1):
            self.tr.train_tools.save_model(self.state, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/', str + '_encoder.pth'))
        else:
            self.tr.train_tools.save_model(self.state, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/', str + '.pth'))
        
    def set_encoder(self):
        if self.state == 1:
            return
        
        if len(self.modality) == 1:
            self.model.encoder.load_state_dict(torch.load(os.path.join( os.path.dirname(os.path.abspath(__file__)),  'models', self.modality[0]+'_encoder.pth'), weights_only=True))
        elif len(self.modality) == 2:
            self.model.encoder.encoder_1.load_state_dict(torch.load(os.path.join( os.path.dirname(os.path.abspath(__file__)),  'models', self.modality[0]+'_encoder.pth'), weights_only=True)) #['model'].encoder
            self.model.encoder.encoder_2.load_state_dict(torch.load(os.path.join( os.path.dirname(os.path.abspath(__file__)),  'models', self.modality[1]+'_encoder.pth'), weights_only=True))
        else:
            self.model.encoder.encoder_1.load_state_dict(torch.load(os.path.join( os.path.dirname(os.path.abspath(__file__)),  'models', self.modality[0]+'_encoder.pth'), weights_only=True))
            self.model.encoder.encoder_2.load_state_dict(torch.load(os.path.join( os.path.dirname(os.path.abspath(__file__)),  'models', self.modality[1]+'_encoder.pth'), weights_only=True))
            self.model.encoder.encoder_3.load_state_dict(torch.load(os.path.join( os.path.dirname(os.path.abspath(__file__)),  'models', self.modality[2]+'_encoder.pth'), weights_only=True))
        