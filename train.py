import torch
import os
import time
import matplotlib.pyplot as plt
from Model.network import LSTMModel
from config import MODEL_CONFIG, TRAINING_SETTINGS
from Utils.plotting import plot_losses
from Utils.training import train
from Data.dataloader import create_data_loaderV1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = os.path.join(os.path.dirname(__file__), "cleaned_data.csv")

def main(model_cfg, training_cfg):
    
    start_time = time.time()
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
        print("\n Opened text file:", f.name)
        print("\n Number of data points:", len(data))
    
    # Initialize model    
    model = LSTMModel(model_config=model_cfg).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg["peak_lr"],
                                 weight_decay=training_cfg["weight_decay"])
    
    # Create dataloader objects
    train_ratio = 0.8 # train/validation split
    split_idx = int(train_ratio * len(data))
    
    train_loader = create_data_loaderV1(
        data[:split_idx],
        batch_size=training_cfg["batch_size"],
        drop_last=False,
        shuffle=False,
    )
    
    val_loader = create_data_loaderV1(
        data[split_idx:],
        batch_size=training_cfg["batch_size"],
        drop_last=False,
        shuffle=False,
    )
    
    # Train model
    train_losses, val_losses = train(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=training_cfg["num_epochs"], eval_freq=5,
        eval_iter=1
    )
    
    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f"Training completed in {execution_time:.2f} minutes.")
    
    return train_losses, val_losses, model

if __name__ == "__main__":
    # Initiate training
    train_losses, val_losess, model = main(MODEL_CONFIG, TRAINING_SETTINGS)
    
    # Plot results
    epochs_tensor = torch.linspace(0, TRAINING_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, train_losses, val_losess)
    plt.savefig("loss.pdf")
    
    # Save and load model
    torch.save(model.state_dict(), "model.pth")
    model = LSTMModel(MODEL_CONFIG).to(device)
    model.load_state_dict(torch.load("model.pth"))