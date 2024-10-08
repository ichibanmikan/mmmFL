import numpy as np

class mmFedAvg:
    def __init__(self):
        pass
    
    def calculate_average(self, arrays):
        stacked_arrays = np.stack(arrays)
        return np.mean(stacked_arrays, axis=0)
