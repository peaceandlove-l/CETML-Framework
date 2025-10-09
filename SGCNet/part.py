import torch.nn as nn
from einops import rearrange
from einops import repeat
import math
import torch
import numpy as np
from torch.nn import init
from itertools import repeat
from torch.nn import functional as F
from torch._jit_internal import Optional
import collections


class DOConv2d(nn.Module):
    """
       DOConv2d can be used as an alternative for torch.nn.Conv2d.
       The interface is similar to that of Conv2d, with one exception:
            1. D_mul: the depth multiplier for the over-parameterization.
       Note that the groups parameter switchs between DO-Conv (groups=1),
       DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size, groups, D_mul=None, stride=1,
                 padding=1, dilation=1, bias=False, padding_mode='zeros'):
        super(DOConv2d, self).__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))

        #################################### Initailization of D & W ###################################
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.W = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if M * N > 1:
            self.D = nn.Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            d_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0: # the cases when D_mul > M * N
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.d_diag = nn.Parameter(torch.cat([d_diag, zeros], dim=2), requires_grad=False)
            else: # the case when D_mul = M * N
                self.d_diag = nn.Parameter(d_diag, requires_grad=False)
        ##################################################################################################

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if M * N > 1:
            ######################### Compute DoW #################
            # (input_channels, D_mul, M * N)
            D = self.D + self.d_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

            # einsum outputs (out_channels // groups, in_channels, M * N),
            # which is reshaped to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
            #######################################################
        else:
            # in this case D_mul == M * N
            # reshape from
            # (out_channels, in_channels // groups, D_mul)
            # to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(self.W, DoW_shape)
        return self._conv_forward(input, DoW)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)

class AutoSelectedSparseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super(AutoSelectedSparseAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = DOConv2d(dim * 3, dim * 3, kernel_size=3, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn1 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k = map(lambda t: F.normalize(t, dim=-1), (q, k))

        _, _, C, _ = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # Generate BOOLEAN masks for different k values
        k_values = [int(C / 2), int(C * 2 / 3), int(C * 3 / 4), int(C * 4 / 5)]
        masks = []
        for k_val in k_values:
            index = torch.topk(attn, k=k_val, dim=-1, largest=True)[1]
            mask = torch.zeros_like(attn, device=x.device, dtype=torch.bool)  # 创建布尔张量
            mask.scatter_(-1, index, True)  # 填充True值
            masks.append(mask)
        masks = torch.stack(masks)

        # Apply masks
        attn = attn.unsqueeze(0)
        attn_masked = torch.where(masks, attn, torch.full_like(attn, float('-inf')))
        attn_masked = F.softmax(attn_masked, dim=-1)

        # Batch matrix multiplication
        v = v.unsqueeze(0)  # (1, b, h, C, L)
        out = torch.matmul(attn_masked, v)  # (4, b, h, C, L)

        # Weighted sum
        weights = torch.stack([self.attn1, self.attn2, self.attn3, self.attn4])
        weights = weights.view(4, 1, 1, 1, 1)  # Match dimensions
        out = (out * weights).sum(dim=0)

        out = rearrange(out, 'b h c (h_w w_w) -> b (h c) h_w w_w', h_w=h, w_w=w)
        return self.project_out(out)



class ChannelOrthogonalFilterAttention(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width

        # 确保降维后的通道数不小于1
        self.reduced_dim = max(channels // 16, 1)

        # 初始化滤波器时不需要指定设备
        self.register_buffer('_gram_schmidt_filter', initialize_orthogonal_filters(channels, height, width))

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, self.reduced_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reduced_dim, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        if (H, W) != (self.height, self.width):
            x = F.adaptive_avg_pool2d(x, (self.height, self.width))

        # 动态对齐设备（关键修改）
        gram_filter = self._gram_schmidt_filter.to(x.device)

        transformed = (gram_filter * x).sum(dim=(2, 3), keepdim=True)
        excitation = self.channel_attention(transformed)
        return x * excitation

def gram_schmidt(input):
    def projection(u, v):
        return (torch.dot(u.view(-1), v.view(-1)) / torch.dot(u.view(-1), u.view(-1))) * u

    output = []
    for x in input:
        for y in output:
            x = x - projection(y, x)
        x = x / x.norm(p=2)
        output.append(x)
    return torch.stack(output)


def initialize_orthogonal_filters(c, h, w):
    total_elements = c * h * w
    if h * w < c:
        filters = torch.rand(c, h * w)
        orthogonal_filters = gram_schmidt(filters).view(c, h, w)
    else:
        filters = torch.rand(c, h * w)
        orthogonal_filters = gram_schmidt(filters).view(c, h, w)
    return orthogonal_filters


class GramSchmidtTransform(nn.Module):
    instance = {}

    @staticmethod
    def build(c, h, w):
        if (c, h, w) not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h, w)] = GramSchmidtTransform(c, h, w)
        return GramSchmidtTransform.instance[(c, h, w)]

    def __init__(self, c, h, w):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h, w)
        self.register_buffer("constant_filter", rand_ortho_filters.to(self.device).detach())

    def forward(self, x):
        _, _, h, w = x.shape
        _, H, W = self.constant_filter.shape
        if h != H or w != W:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (H, W))
        return (self.constant_filter * x).sum(dim=(-1, -2), keepdim=True)

if __name__ == '__main__':
    rgb = torch.randn(2, 64, 30, 40)
    d = torch.randn([2, 8, 30, 40])
    net = Top_K_Sparse_Attention(8)
    net = OrthogonalChannelAttention(64,30,40)
    out = net(rgb)
    print("s.shape:", out[-1].shape)