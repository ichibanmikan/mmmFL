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
import numpy as np
from global_models.Flash_model import Flash
from global_models.MHAD_model import MHAD

class globel_models_manager:
    def __init__(self):
        self.models = []
        self.models.append(MHAD())
        self.models.append(Flash())
        
    def get_model_params(self, task):
        return self.models[task].get_model_params()
        
    def reset_models(self, task, new_params_vec):
        new_params = np.mean(new_params_vec, axis=0)
        params_init = self.models[task].get_model_params()
        self.models[task].reset_model_parameter(params_init+new_params)
    
    def get_model_name(self, task):
        print(task)
        print(self.models[task].get_model_name())
        