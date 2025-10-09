import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from collections import OrderedDict
import math
from mymodelnew.Backbone.segformer.mix_transformer import mit_b2
# from mymodelnew.toolbox.models.segformer.mix_transformer import mit_b3
# from mymodelnew.toolbox.models.segformer.mix_transformer import mit_b4
# from mymodelnew.toolbox.models.segformer.mix_transformer import mit_b5
# from mymodelnew.toolbox.Mymodels.Baseline_lsy_seg.NEW import UFFConv
from mymodelnew.toolbox.Mymodels.Baseline_lsy_seg.MLPDecoder import DecoderHead


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class FourierUnit(nn.Module):
    def __init__(self, dim, groups=1, fft_norm='ortho'):
        super().__init__()
        self.groups = groups
        self.fft_norm = fft_norm

        self.conv_layer = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, stride=1,
                                    padding=0, groups=self.groups, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        # ffted = torch.fft.rfft(x, signal_ndim=2, normalized=True)
        ffted = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')

        ffted = torch.stack([ffted.real, ffted.imag], dim=-1)

        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act(ffted)

        # (batch, c, h, w/2+1, 2)
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()

        # 将实虚部分分离的格式转换为复数张量
        complex_tensor = torch.complex(ffted[..., 0], ffted[..., 1])

        # 使用irfft2替代irfft，并使用新版参数
        output = torch.fft.irfft2(
            complex_tensor,
            s=r_size[2:],  # 指定输出尺寸
            dim=(-2, -1),  # 指定进行逆FFT的维度
            norm='ortho'  # 指定归一化方式
        )

        return output


class FConvMod(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        layer_scale_init_value = 1e-6
        self.num_heads = num_heads
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = FourierUnit(dim)
        self.v = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(num_heads), requires_grad=True)
        self.CPE = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        shortcut = x
        pos_embed = self.CPE(x)
        x = self.norm(x)
        a = self.a(x)
        v = self.v(x)
        a = rearrange(a, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        a_all = torch.split(a, math.ceil(N // 4), dim=-1)
        v_all = torch.split(v, math.ceil(N // 4), dim=-1)
        attns = []
        for a, v in zip(a_all, v_all):
            attn = a * v
            attn = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * attn
            attns.append(attn)
        x = torch.cat(attns, dim=-1)
        x = F.softmax(x, dim=-1)
        x = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        x = x + pos_embed
        x = self.proj(x)
        out = x + shortcut

        return out


class KernelAggregation(nn.Module):
    def __init__(self, dim, kernel_size, groups, num_kernels, bias=True, init_weight=True):
        super().__init__()
        self.groups = groups
        self.bias = bias
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.dim = dim
        self.weight = nn.Parameter(torch.randn(num_kernels, dim, dim // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_kernels, dim))
        else:
            self.bias = None

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.num_kernels):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x, attention):
        B, C, H, W = x.shape
        x = x.contiguous().view(1, B * self.dim, H, W)

        weight = self.weight.contiguous().view(self.num_kernels, -1)
        weight = torch.mm(attention, weight).contiguous().view(B * self.dim, self.dim // self.groups,
                                                               self.kernel_size, self.kernel_size)
        if self.bias is not None:
            bias = torch.mm(attention, self.bias).contiguous().view(-1)
            x = F.conv2d(x, weight=weight, bias=bias, stride=1, padding=self.kernel_size // 2,
                         groups=self.groups * B)
        else:
            x = F.conv2d(x, weight=weight, bias=None, stride=1, padding=self.kernel_size // 2,
                         groups=self.groups * B)
        x = x.contiguous().view(B, self.dim, x.shape[-2], x.shape[-1])

        return x


class KernelAttention(nn.Module):
    def __init__(self, dim, reduction=8, num_kernels=8):
        super().__init__()
        if dim != 3:
            mid_channels = dim // reduction
        else:
            mid_channels = num_kernels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, mid_channels, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channels, num_kernels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.sigmoid(x)
        return x


class DynamicKernelAggregation(nn.Module):
    def __init__(self, dim, kernel_size, groups=1, num_kernels=4):
        super().__init__()
        assert dim % groups == 0
        self.attention = KernelAttention(dim, num_kernels=num_kernels)
        self.aggregation = KernelAggregation(dim, kernel_size=kernel_size, groups=groups, num_kernels=num_kernels)

    def forward(self, x):
        attention = x
        attention = self.attention(attention)
        x = self.aggregation(x, attention)
        return x


class DyConv(nn.Module):
    def __init__(self, dim, kernel_size, groups, num_kernels=1):
        super().__init__()
        if num_kernels > 1:
            self.conv = DynamicKernelAggregation(dim, kernel_size=kernel_size, groups=groups,
                                                 num_kernels=num_kernels)
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=groups)

    def forward(self, x):
        x = self.conv(x)
        return x


class MixFFN(nn.Module):
    def __init__(self, dim, num_kernels):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, dim * 2, 1)
        self.conv1 = DyConv(dim, kernel_size=5, groups=dim, num_kernels=num_kernels)
        self.conv2 = DyConv(dim, kernel_size=7, groups=dim, num_kernels=num_kernels)
        self.proj_out = nn.Conv2d(dim * 2, dim, 1)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.act(self.proj_in(x))
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.act(self.conv1(x1)).unsqueeze(dim=2)
        x2 = self.act(self.conv2(x2)).unsqueeze(dim=2)
        x = torch.cat([x1, x2], dim=2)
        x = rearrange(x, 'b c g h w -> b (c g) h w')
        x = self.proj_out(x)
        x = x + shortcut
        return x


class BasicBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kernels):
        super().__init__()
        self.attention = FConvMod(dim, num_heads)
        self.ffn = MixFFN(dim, num_kernels)

    def forward(self, x):
        x = self.attention(x)
        x = self.ffn(x)

        return x


#########Fusion
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = rearrange(x2,'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
class Residualfrequencyfusion(nn.Module):
    def __init__(self, dim, h, w, reduction=8):
        super(Residualfrequencyfusion, self).__init__()
        self.c = dim
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.FDM = BasicBlock(dim,num_heads=4,num_kernels=4)


        self.h = h
        self.w = w


    def forward(self, x, y):
        y = F.interpolate(y, size=(self.h, self.w), mode='bilinear', align_corners=False)

        initial = x + y
        pattn1 = initial + initial
        # pattn2 = self.sigmoid(self.UFFConv(self.pa(initial, pattn1)))
        pattn2 = self.sigmoid(self.FDM(self.pa(initial, pattn1)))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)

        return result


class Channelselectrelatedfusion(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(Channelselectrelatedfusion, self).__init__()

        # 共享的全连接层（减少重复参数）
        self.shared_fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True)
        )

        # 通道注意力分支
        self.fc_attention = nn.Sequential(
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

        # 融合权重分支
        self.fc_weight = nn.Sequential(
            nn.Linear(ch_in // reduction, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_h, x_l):
        b, c, h, w = x_h.size()
        # 并行计算高低层次特征的全局信息
        y_h = x_h.mean(dim=(2, 3))  # [b,c]
        y_l = x_l.mean(dim=(2, 3))  # [b,c]
        combined = torch.cat([y_h, y_l], dim=0)  # [2b,c]
        # 通过共享层处理
        shared_out = self.shared_fc(combined)  # [2b, c//r]
        # 并行计算注意力权重和融合权重
        att = self.fc_attention(shared_out)  # [2b,c]
        weights = self.fc_weight(shared_out)  # [2b,1]
        # 拆分结果
        att_h, att_l = torch.chunk(att, 2, dim=0)
        w_h, w_l = torch.chunk(weights, 2, dim=0)
        # 应用注意力机制
        x_fusion_h = x_h * att_h.view(b, c, 1, 1) * w_h.view(b, 1, 1, 1)
        x_fusion_l = x_l * att_l.view(b, c, 1, 1) * w_l.view(b, 1, 1, 1)

        return x_fusion_h + x_fusion_l

class Net2(nn.Module):
    def __init__(self,num_class=41,):
        super(Net2,self).__init__()
        self.backbone = mit_b2()
        # self.backbone = p2t_small()

        self.Mlp_decoder = DecoderHead()
        dim = [64,128,160,256]


        self.sig = nn.Sigmoid()

        self.CSRF1 = Channelselectrelatedfusion(dim[0])
        self.CSRF2 = Channelselectrelatedfusion(dim[1])
        self.CSRF3 = Channelselectrelatedfusion(dim[2])
        self.CSRF4 = Channelselectrelatedfusion(dim[3])

        h = [120,60,30,15]
        self.h = h
        w = [160,80,40,20]
        self.w = w

        #feature Enhance
        self.conv64to128 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1)
        self.conv128to160 = nn.Conv2d(in_channels=128,out_channels=160,kernel_size=1)
        self.conv160to256 = nn.Conv2d(in_channels=160,out_channels=256,kernel_size=1)
        self.conv320to160 = nn.Conv2d(in_channels=320,out_channels=160,kernel_size=1)
        self.conv512to256 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1)
        self.RFF1 = Residualfrequencyfusion(dim[0],h[0],w[0])
        self.RFF2 = Residualfrequencyfusion(dim[1],h[1],w[1])
        self.RFF3 = Residualfrequencyfusion(dim[2],h[2],w[2])
        self.RFF4 = Residualfrequencyfusion(dim[3],h[3],w[3])


        self.convdiff = nn.Conv2d(64,256,1)



    def forward(self, rgb, dep):
        rgb = self.backbone(rgb)
        depth = self.backbone(dep)
        rgb[2] = self.conv320to160(rgb[2])
        depth[2] = self.conv320to160(depth[2])
        rgb[3] = self.conv512to256(rgb[3])
        depth[3] = self.conv512to256(depth[3])


        fuse0 = self.CSRF1(rgb[0], depth[0])
        fuse1 = self.CSRF2(rgb[1], depth[1])
        fuse2 = self.CSRF3(rgb[2], depth[2])
        fuse3 = self.CSRF4(rgb[3], depth[3])


        fuse0 = self.RFF1(fuse0,fuse0)
        x0 = self.conv64to128(fuse0)
        fuse1 = self.RFF2(fuse1,x0)
        x1 = self.conv128to160(fuse1)
        fuse2 = self.RFF3(fuse2,x1)
        x2 = self.conv160to256(fuse2)
        fuse3 = self.RFF4(fuse3,x2)

        ##mutal
        x3 = F.interpolate(fuse3,size=(self.h[0],self.w[0]),mode='bilinear',align_corners=False)
        x2 = F.interpolate(fuse2,size=(self.h[0],self.w[0]),mode='bilinear',align_corners=False)
        x1 = F.interpolate(fuse1,size=(self.h[0],self.w[0]),mode='bilinear',align_corners=False)
        x0 = self.convdiff(fuse0)



        out = self.Mlp_decoder(fuse0, fuse1, fuse2, fuse3)


        return out,x0,x1,x2,x3

    def load_pre_sa(self, pre_model):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model)['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        self.backbone.load_state_dict(new_state_dict3, strict=False)
        print('self.backbone_seg_mit loading')



if __name__ == '__main__':
    net = Net2().cuda()
    rgb = torch.randn([1, 3, 480, 640]).cuda()
    d = torch.randn([1, 3, 480, 640]).cuda()
    s = net(rgb, d)
    print("==> Total params: %.2fM" % (sum(p.numel() for p in net.parameters()) / 1e6))
    from mymodelnew.toolbox.Mymodels.Baseline_lsy_seg.FLOP import CalParams
    CalParams(net, rgb, d)
    # print("s.shape:", s[0][-1].shape)
    # print("s.shape:", s[1][-1].shape)
    # print("s.shape:", s[2][-1].shape)
    print("s.shape:", s[1][-1].shape)
    print("s.shape:", s[4][-1].shape)