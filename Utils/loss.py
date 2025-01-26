import torch
import math

# Calculate loss per batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    predictions = model(input_batch)
    return torch.nn.functional.mse_loss(predictions.flatten(0, 1),
                                        target_batch.flatten())
    
# Dataloading for loss calculations
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches is len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
        
    return total_loss / num_batches

# Model Evaluation
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device,
                                      num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device,
                                    num_batches=eval_iter)
    model.train()
    
    return train_loss, val_loss