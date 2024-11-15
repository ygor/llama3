import torch
from torch import nn
from model_args import ModelArgs


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        device = ModelArgs.device
        self.eps = eps
        # Scaling parameter gamma, initialized with one and the no of parameters is equal to the size of dim
        self.weight = nn.Parameter(torch.ones(dim).to(device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(
            device
        )

    def forward(self, x):
        # Shape: x[bs,seq,dim]
        output = self._norm(x.float()).type_as(x)

        # Shape: x[bs,seq,dim] -> x_norm[bs,seq,dim]
        return output * self.weight


### Test: RMSNorm Code ###
# You need take out the triple quotes below to perform testing
"""
x = torch.randn((ModelArgs.max_batch_size, ModelArgs.max_seq_len, ModelArgs.dim), device=device)
rms_norm = RMSNorm(dim=ModelArgs.dim)
x_norm = rms_norm(x)

print(f"Shape of x: {x.shape}")
print(f"Shape of x_norm: {x_norm.shape}")
"""
### Test Results: ###
"""
Shape of x: torch.Size([10, 256, 512])
Shape of x_norm: torch.Size([10, 256, 512])
"""
