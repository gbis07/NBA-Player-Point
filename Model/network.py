import torch
import torch.nn as nn
from Model.layers import NormLayer, MHA, FeedForward

class Block(nn.Module):
    # replace input and output with config argument
    def __init__(self, model_config):
        super(Block, self).__init__()
        self.att = MHA(
            input_dim=model_config["input_dim"],
            output_dim=model_config["output_dim"],
            num_heads=model_config["num_heads"],
            dropout=model_config["dropout"],
            qkv_bias=model_config.get("qkv_bias", False)
        )
        # self.ff = FeedForward(model_config)
        # self.norm1 = NormLayer(model_config["input_dim"])
        # self.norm3 = NormLayer(model_config["input-dim"])
        self.fc1 = nn.Linear(model_config["input_dim"], model_config["input_dim"]//2)
        # self.norm2 = NormLayer(model_config["input_dim"]//2)
        # self.norm4 = NormLayer(model_config["input_dim"]//2)
        self.fc2 = nn.Linear(model_config["input_dim"]//2, model_config["output_dim"])
        self.fc3 = nn.Linear(model_config["input_dim"], model_config["output_dim"])
        self.drop_shortcut = nn.Dropout(model_config["drop_rate"])
        
    def forward(self, x):
        # Shortcut for attention block
        shortcut = x
        # x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        skip = self.fc3(x)
        # x = self.norm2(self.fc1(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = skip + shortcut + x
    
        # Shortcut
        shortcut = x
        # x = self.norm3(x)
        # x = self.ff(x)
        # x = self.drop_shortcut(x)
        # skip = self.fc3(x)
        # x = self.norm4(self.fc1(x))
        # x = self.fc2
        # x = skip + shortcut + x
        
        return x 
    
class LSTMModel(nn.Module):
    # replace input, output, etc... with model config
    def __init__(self, model_config, biredctional=False):
        super(LSTMModel, self).__init__()
        self.hidden = model_config["hidden_size"]
        self.num_heads = model_config["num_heads"]
        self.direction = 2 if biredctional else 1
        self.lstm = nn.LSTM(input_size=model_config["input_dim"],
                            hidden_size=model_config["hidden_size"],
                            num_layers=model_config["num_layers"],
                            batch_first=True,
                            bidirectional=biredctional)
        self.blocks = nn.Sequential(*[Block(model_config) for _ in range(model_config["num_blocks"])])
        self.fc_out = nn.Linear(model_config["hidden_size"]*self.direction, model_config["output_dim"])
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.blocks(lstm_out)
        out = self.fc_out(attn_out)
        
        return out