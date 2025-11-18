import json
import os
from MHAD.data import *
from MHAD.worker import *

class Config:
    def __init__(self, config_path) -> None:
        self.config_path = config_path
        self.load_config()

    def load_config(self) -> None:
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)
        self.batch_size = config_data.get('batch_size', 16)
        self.num_workers = config_data.get('num_workers', 16)
        self.epochs = config_data.get('epochs', 99)
        self.learning_rate = config_data.get('learning_rate', 0.001)
        self.lr_decay_rate = config_data.get('lr_decay_rate', 0.9)
        self.weight_decay = config_data.get('weight_decay', 0.0001)
        self.momentum = config_data.get('momentum', 0.9)
        self.num_classes = config_data.get('num_classes', 11)
        self.total_epochs = config_data.get('total_epochs', 200)
        self.MACs = config_data.get('MACs', 1)

    def __repr__(self) -> str:
        return f"Config({self.__dict__})"

class MHAD_main:
    def __init__(self, modality, node_id, modal_size):
        self.modality = modality
        self.now_loss = 999
        self.config = Config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'))
        self.modal_size = modal_size
        self.model = MyMMModel(self.config.num_classes)      

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.model = self.model.to(device)
        data_f = data_factory(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/node_'+f"{node_id}/"), self.config)
        train_loader = data_f.get_dataset()
        self.tr = Trainer(self.config, self.model, train_loader, device)
        self.node_id = node_id
        self.MACs = data_f.sample_length * self.config.MACs
        
    def main(self):
        self.now_loss = self.tr.train()
        # print(self.tr.best_acc)
        
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
        # print("Shape of model weight: ", model_params.shape)

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
        
    

# The distance between the clients and the BS follows a uniform distribution di ∼ Uniform(1, 100) in
# meters. The wireless transmission powers of the clients are 10 − 20 dBm. The communication bandwidth
# between the BS and the clients is 20 MHz, and the Gaussian noise power is around −101 dBm. We
# adopt long-distance path loss model to calculate the channel gain for each client. Specifically, we have
# PL(di) = 40 + 30 log10 di + ϱ (in dB) where ϱ ∼ N (0, 62), and gi = 10−PL(di)/10.
# We assume the MAC rate of each client i ∈ N , i.e., κi, is in the range [5 × 107, 3 × 108] MAC/s,
# and the energy consumption of each client i to performance one MAC operation, i.e., ρi, is in the range
# [0.1, 1] pJ/MAC. Let the energy budgets of the clients be in the range [xx, xx] J for both communications
# and computations.