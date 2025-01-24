import torch
from torch.utils.data import Dataset, DataLoader

class DatasetV1(Dataset):
    
    # Constructor
    def __init__(self, data, targets, transform=None):
        self.x = data
        self.y = targets
        self.transform = transform
        
        if self.transform:
            x = self.transform(x)
    
    # Get Item at Index    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    # Get Length
    def __len__(self):
        return len(self.x)
    
def create_data_loaderV1(data, targets, batch_size=4, shuffle=True):
    dataset = DatasetV1(data, targets)
    dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
