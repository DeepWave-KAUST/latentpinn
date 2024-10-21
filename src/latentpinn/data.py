import torch
import torch.nn
import torch.utils.data
import numpy as np
import os.path
import multiprocessing
import timeit
import time

from torch.utils.data import Dataset, DataLoader

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom Dataset class
class NumpyDataset(Dataset):
    def __init__(self, data, normalize=None, water_velocity=1.5, water_depth=0, device='cuda', num_samples=36000):
        
        if water_depth != 0:
            data[:,:water_depth,:] = water_velocity
            
        self.max_velocity = torch.max(data)    
        self.min_velocity = torch.min(data)      
        
        if normalize:
            # self.data = self._normalize(data[:,::2,::2]) # Transform Reseam from 256 to 128
            self.data = self._normalize(data) # Transform Reseam from 256 to 128
        else:
            # self.data = data[:,::2,::2] 
            self.data = data 
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return sample.reshape(-1,128,128).float()

    def _normalize(self, array):
        normalized_array = 2 * (array - self.min_velocity) / (self.max_velocity - self.min_velocity) - 1
        return normalized_array