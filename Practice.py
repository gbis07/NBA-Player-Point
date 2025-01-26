import torch
from Data.dataloader import create_data_loaderV1

num_smaples = 1000
seq_length = 20
input_size = 10
output_size = 8
batch_size = 4

data = torch.randn(num_smaples, seq_length, input_size)
targets = torch.randn(num_smaples, output_size)

data_loader = create_data_loaderV1(data, targets, batch_size=batch_size)

for batch_idx, (x_batch, y_batch) in enumerate(data_loader):
    print(f"Batch {batch_idx + 1}")
    print(f"Input shape: {x_batch.shape}")
    print(f"Output shape: {y_batch.shape}")