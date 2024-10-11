import numpy as np

class mmFedAvg:
    def __init__(self):
        self.global_encoder = {        
            "text": np.zeros(1000),
            "audio": np.zeros(600),
            "video": np.zeros(1500)
        }
    
    def calculate_average(self, arrays, modality):
        stacked_arrays = np.stack(arrays)
        self.global_encoder[modality] = np.mean(stacked_arrays, axis=0)

    def __getitem__(self, key):
        return self.global_encoder[key]

    def __setitem__(self, key, value):
        self.global_encoder[key] = value        
