import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# relative positional embedding

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

# classes

class HaloAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        block_size,
        halo_size,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.block_size = block_size
        self.halo_size = halo_size

        inner_dim = dim_head * heads

        self.rel_pos_emb = RelPosEmb(
            block_size = block_size,
            rel_size = block_size + (halo_size * 2),
            dim_head = dim_head
        )

        self.to_q  = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, c, h, w, block, halo, heads, device = *x.shape, self.block_size, self.halo_size, self.heads, x.device
        assert h % block == 0 and w % block == 0, 'fmap dimensions must be divisible by the block size'
        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'

        # get block neighborhoods, and prepare a halo-ed version (blocks with padding) for deriving key values

        q_inp = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = block, p2 = block)

        kv_inp = F.unfold(x, kernel_size = block + halo * 2, stride = block, padding = halo)
        kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c = c)

        # derive queries, keys, values

        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = heads), (q, k, v))

        # scale

        q *= self.scale

        # attention

        sim = einsum('b i d, b j d -> b i j', q, k)

        # add relative positional bias

        sim += self.rel_pos_emb(q)

        # mask out padding (in the paper, they claim to not need masks, but what about padding?)

        mask = torch.ones(1, 1, h, w, device = device)
        mask = F.unfold(mask, kernel_size = block + (halo * 2), stride = block, padding = halo)
        mask = repeat(mask, '() j i -> (b i h) () j', b = b, h = heads)
        mask = mask.bool()

        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1)

        # aggregate

        out = einsum('b i j, b j d -> b i d', attn, v)

        # merge and combine heads

        out = rearrange(out, '(b h) n d -> b n (h d)', h = heads)
        out = self.to_out(out)

        # merge blocks back to original feature map

        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b = b, h = (h // block), w = (w // block), p1 = block, p2 = block)
        return out


class HaloBlock(nn.Module):
    def __init__(self, dim, block_size, halo_size, dim_head, heads):
        super(HaloBlock, self).__init__()
        self.attn1 = HaloAttention(dim = dim, block_size = block_size, halo_size = halo_size, dim_head = dim_head, heads = heads)
        self.in1 = nn.InstanceNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.attn2 = HaloAttention(dim = dim, block_size = block_size, halo_size = halo_size, dim_head = dim_head, heads = heads)
        self.in2 = nn.InstanceNorm2d(dim)

    def forward(self, x):
        residual = x

        out = self.attn1(x)
        out = self.in1(out)
        out = self.relu(out)

        out = self.attn2(out)
        out = self.in2(out)

        out += residual
        out = self.relu(out)

        return out


class HaloNet(nn.Module):
    def __init__(self, args = {}):
        super(HaloNet, self).__init__()
        in_channel, out_channel = args.get('input_channel', 441), args.get('out_channel', 10)
        intermediate_channel = args.get('intermediate_channel', 64)
        blocks = args.get('blocks', 22)
        block_size = args.get('block_size', 8)
        halo_size = args.get('halo_size', 4)
        dim_head = args.get('dim_head', 16)
        heads = args.get('heads', 4)

        self.conv1 = nn.Conv2d(in_channel, intermediate_channel, kernel_size=1, bias=False)
        self.in1 = nn.InstanceNorm2d(intermediate_channel)
        self.relu = nn.ReLU(inplace=True)

        layer = []
        for i in range(0, int(blocks)):
            layer.append(HaloBlock(intermediate_channel, block_size, halo_size, dim_head, heads))
        self.layer = nn.Sequential(*layer)

        self.final = nn.Conv2d(intermediate_channel, out_channel, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        x = self.relu(self.in1(self.conv1(x)))
        x = self.layer(x)
        x = self.final(x)

        return x
