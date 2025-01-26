MODEL_CONFIG = {
    "input_dim": 30,
    "output_dim": 8,
    "num_heads": 4,
    "hidden_size": 12,
    "num_layers" : 1,
    "mnum_blocks": 1,
    "drop_rate": 0.1,
    "qkv_bias": False
}

TRAINING_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 10,
    "batch_size": 2,
    "weight_decay": 0.1,
    "peak_lr": 0.001,
}