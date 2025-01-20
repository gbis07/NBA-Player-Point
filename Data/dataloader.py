import torch
from torch.utils.data import Dataset, DataLoader

class DatasetV1(Dataset):
    
    # Constructor
    def __init__(self, length=100, transform= None):
        pass
    
    # Get Item at Index    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len
    
def create_data_loaderV1():
    pass
