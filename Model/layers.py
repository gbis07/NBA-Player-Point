import torch
import torch.nn as nn

# class NormLayer(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.eps = 1e-5
#         self.scale = nn.Parameter(torch.ones(input_dim))
#         self.shift = nn.Parameter(torch.zeros(input_dim))
        
#     def forward(self, x):
#         mean = x.mean(dim=-1, keepdim=True)
#         var = x.var(dim=-1, unbiased=False)
#         norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
#         return self.scake * norm_x + self.shift
    
# class FeedForward(nn.Module):
#     def __init__(self, model_config):
#         super().__init()
#         self.layers = nn.Sequential(nn.Linear(model_config["input_dim"], model_config["input_dim"]))
    
#     def forward(self, x):
#         return self.layers(x)
        
class MHA(nn.Module):
    def __init__(self, input_dim, hidden_size, 
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert hidden_size % num_heads == 0, "Number of hidden layers must be divisible by number of attnention heads"
        self.num_heads = num_heads
        self.head_d = hidden_size // num_heads
        self.hidd = hidden_size
        
        # Separate Linear layers for query, key, and value
        self.qkv = nn.ModuleList([nn.Linear(input_dim, hidden_size, bias=qkv_bias) for _ in range(3)])
        self.proj = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, lstm_output):
        batch_size, seq_length, _ = lstm_output.size()
        
        # Generate query, key, and value vectors from separate layers
        queries = self.qkv[0](lstm_output).view(batch_size, seq_length, self.num_heads, self.head_d)
        keys = self.qkv[1](lstm_output).view(batch_size, seq_length, self.num_heads, self.head_d)
        values = self.qkv[2](lstm_output).view(batch_size, seq_length, self.num_heads, self.head_d)
        
        # Permute for attention computation: (batch_size, num_heads, seq_length, head_d)
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        
        # Used Pytorch Scaled Dot-Product-Attention
        context_vector = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=self.drop, is_causal=True)
        
        # Combine heads and project output
        context_vector = context_vector.transpose(1, 2).contigous().view(batch_size, seq_length, self.hidd)
        context_vector =self.proj(context_vector)
        
        return context_vector