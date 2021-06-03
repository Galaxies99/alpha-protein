import torch
from models.halonet import HaloAttention

attn = HaloAttention(
    dim = 441,         # dimension of feature map
    block_size = 8,    # neighborhood block size (feature map must be divisible by this)
    halo_size = 4,     # halo size (block receptive field)
    dim_head = 64,     # dimension of each head
    heads = 4          # number of attention heads
)

fmap = torch.randn(1, 441, 32, 32)
print(attn(fmap).shape)

fmap = torch.randn(1, 441, 64, 64)
print(attn(fmap).shape)