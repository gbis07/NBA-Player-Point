import torch
from Utils.loss import calc_loss_batch, evaluate_model

def train(model, train_loader, val_loader, optimizer, 
          device, num_epochs, eval_freq, eval_iter):
    # Lists for losses 
    train_losses, val_losses = [], []
    global_step = -1
    
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Resets gradients from previous iterations
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculates loss and backpropagates to parameters
            optimizer.step() # Updates parameters with loss gradients
            global_step += 1
            
            # Evaluation
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader,
                                                      val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f" Epcoch {epoch + 1} (Step {global_step:06d}): "
                      f"Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")
    
    return train_losses, val_losses