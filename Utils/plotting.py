import matplotlib.pyplot as plt

def plot_losses(epochs_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()
    
    ax1.plot(epochs_seen, train_losses, label="Training Loss")
    ax1.plot(epochs_seen, val_losses, label="Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    
    fig.tight_layout()
    plt.show