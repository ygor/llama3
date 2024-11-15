import torch
from torch import nn
from model_args import ModelArgs
from typing import Tuple


class Rope(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    ## Step2b: The RoPE
    def precompute_freqs_cis(self, dim: int, seq_len: int, theta: float = 10000.0):
        # Computing Theta value for each dim pair which is dim/2
        freqs = 1.0 / (
            theta
            ** (torch.arange(0, dim, 2, device=self.device)[: (dim // 2)].float() / dim)
        )

        # Computing range of positions(m) in the sequence
        t = torch.arange(seq_len, dtype=torch.float32, device=self.device)

        # freqs gives all the Theta value range for all the position of tokens in the sequence
        freqs = torch.outer(t, freqs).to(self.device)

        # This is the rotation matrix which needs to be converted to Polar form in order to perform rotation to the embedding
        freqs_cis = torch.polar(torch.ones_like(freqs).to(self.device), freqs).to(
            self.device
        )
        return freqs_cis

    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (
            x.shape[1],
            x.shape[-1],
        ), "the last two dimension of freqs_cis, x must match"
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def apply_rotary_emb(
        self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Applying rotary positional encoding to both query and key embedding together
        # First: The last dimension of xq and xk embedding needs to be reshaped to make it a pair. As rotation matrix is applied to each pair of dim.
        # Next: convert both xq and xk to complex number as the rotation matrix is only applicable to complex number
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)).to(
            self.device
        )  # xq_:[bsz, seq_len, n_heads, head_dim/2]
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)).to(
            self.device
        )  # xk_:[bsz, seq_len, n_heads, head_dim/2]

        # The rotation matrix(freqs_cis) dimensions across seq_len(dim=1) and head_dim(dim=3) should match with the embedding
        # Also, the shape freqs_cis should be the same with xq and xk, hence change the shape of freqs_cis:[seq_len,head_dim] -> freqs_cis:[1,seq_len,1,head_dim]
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)

        # Finally, perform rotation operation by multiplying with freqs_cis.
        # After the rotation is completed, convert both xq_out and xk_out back to real number and return
        xq_out = (
            torch.view_as_real(xq_ * freqs_cis).flatten(3).to(self.device)
        )  # xq_out:[bsz, seq_len, n_heads, head_dim]
        xk_out = (
            torch.view_as_real(xk_ * freqs_cis).flatten(3).to(self.device)
        )  # xk_out:[bsz, seq_len, n_heads, head_dim]
        return xq_out.type_as(xq), xk_out.type_as(xk)

    ### Test: RoPE Code ###
    # Note: x_norm is calculated during RMSNorm and is being used for testing here.
    # You need take out the triple quotes below to perform testing
    """
    head_dim = ModelArgs.dim//ModelArgs.n_heads
    wq = nn.Linear(ModelArgs.dim, ModelArgs.n_heads * head_dim, bias=False, device=device)
    wk = nn.Linear(ModelArgs.dim, ModelArgs.n_kv_heads * head_dim, bias=False, device=device)
    xq = wq(x_norm)
    xk = wk(x_norm)
    print(f"xq.shape: {xq.shape}")
    print(f"xk.shape: {xk.shape}")

    xq = xq.view(xq.shape[0],xq.shape[1],ModelArgs.n_heads, head_dim)
    xk = xk.view(xk.shape[0],xk.shape[1],ModelArgs.n_kv_heads, head_dim)
    print(f"xq.re-shape: {xq.shape}")
    print(f"xk.re-shape: {xk.shape}")

    freqs_cis = precompute_freqs_cis(dim=head_dim, seq_len=ModelArgs.max_seq_len)
    print(f"freqs_cis.shape: {freqs_cis.shape}")

    xq_rotate, xk_rotate = apply_rotary_emb(xq, xk, freqs_cis)
    print(f"xq_rotate.shape: {xq_rotate.shape}")
    print(f"xk_rotate.shape: {xk_rotate.shape}")
    """
    ### Test Results: ###
    """
    xq.shape: torch.Size([10, 256, 512])
    xk.shape: torch.Size([10, 256, 256])
    xq.re-shape: torch.Size([10, 256, 8, 64])
    xk.re-shape: torch.Size([10, 256, 4, 64])
    freqs_cis.shape: torch.Size([256, 32])
    xq_rotate.shape: torch.Size([10, 256, 8, 64])
    xk_rotate.shape: torch.Size([10, 256, 4, 64])
    """
