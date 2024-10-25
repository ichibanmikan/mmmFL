import numpy as np

class mmFedAvg:
    def __init__(self):
        self.global_encoder = {}
    
    def calculate_average(self, arrays, modality):
        new_params = self.global_encoder[self.modalities_key(modality)]
        update = np.zeros(len(arrays[0]))
        for i in range(len(arrays)):
            update += arrays[i]
        update = update / len(arrays)
        return new_params + update
    
    def add_encoder_average(self, array, modality):
        for i in range(len(array)):
            self.global_encoder[self.modalities_key(modality)][i] = (self.global_encoder[self.modalities_key(modality)][i]+array[i]) / 2     
 
        
    def update_param(self, encoder_update, modality):
        str = self.modalities_key(modality)
        if str not in self.global_encoder:
            self.global_encoder[str] = encoder_update
        else:
            self.add_encoder_average(encoder_update, modality)

    def modalities_key(self, modality):
        str = ''
        for i in range(len(modality)):
            str+=modality[i]
        return str

    def __getitem__(self, key):
        str = self.modalities_key(key)
        return self.global_encoder[str]

    def __setitem__(self, key, value):
        str = self.modalities_key(key)
        self.global_encoder[str] = value        
