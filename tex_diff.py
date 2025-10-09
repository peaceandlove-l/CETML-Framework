import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveDiffusion(nn.Module):
    def __init__(self, in_channels, diffusion_steps=3):
        super().__init__()
        self.diffusion_steps = diffusion_steps

        # 渐进式扩散网络
        self.diff_blocks = nn.ModuleList([
            self._make_progressive_block(in_channels, step)
            for step in range(diffusion_steps)
        ])

        # 自适应边缘检测
        self.edge_detector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

        # 注册Sobel滤波器核
        self.register_buffer('sobel_kernel_x', self._get_sobel_kernel(3, 'x'))
        self.register_buffer('sobel_kernel_y', self._get_sobel_kernel(3, 'y'))

    @staticmethod
    def _get_sobel_kernel(kernel_size=3, direction='y'):
        kernel = torch.zeros((1, 1, kernel_size, kernel_size))
        center = kernel_size // 2
        if direction == 'x':
            kernel[0, 0, :, 0] = -1
            kernel[0, 0, :, 2] = 1
            kernel[0, 0, center, 0] = -2
            kernel[0, 0, center, 2] = 2
        elif direction == 'y':
            kernel[0, 0, 0, :] = -1
            kernel[0, 0, 2, :] = 1
            kernel[0, 0, 0, center] = -2
            kernel[0, 0, 2, center] = 2
        return kernel / 8.0

    def _sobel(self, x):
        b, c, h, w = x.shape
        # print('1')
        # 修改1：保持原始通道维度
        grad_x = F.conv2d(
            x,  # 输入保持 [b, c, h, w]
            self.sobel_kernel_x.repeat(c, 1, 1, 1),  # 核形状变为 [c, 1, 3, 3]
            padding=1,
            groups=c  # 关键修改：分组数等于通道数
        )

        grad_y = F.conv2d(
            x,
            self.sobel_kernel_y.repeat(c, 1, 1, 1),
            padding=1,
            groups=c
        )

        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def _make_progressive_block(self, channels, step):
        """创建扩散块（保持输入尺寸不变）"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3 + 2 * step,
                      padding=1 + step, stride=1),  # 保持尺寸
            nn.GroupNorm(4, channels),
            nn.GELU(),
            ChannelAttention(channels)
        )

    def forward(self, x, teacher_feat):
        # 输入形状验证
        assert x.shape == teacher_feat.shape, \
            f"输入形状不匹配: 学生{x.shape} vs 教师{teacher_feat.shape}"

        # 边缘增强
        edge_weight = self.edge_detector(teacher_feat)
        x = x * (1 + edge_weight)

        # 多步骤扩散处理
        for block in self.diff_blocks:
            x = block(x)

        # 计算损失
        loss = self._calc_loss(x, teacher_feat)
        return loss

    def _calc_loss(self, student_feat, teacher_feat):
        # 自动对齐特征尺寸（安全机制）
        if teacher_feat.shape[-2:] != student_feat.shape[-2:]:
            teacher_feat = F.interpolate(teacher_feat,size=student_feat.shape[-2:],mode='bilinear')

        # 风格损失
        style_loss = F.l1_loss(self._gram_matrix(student_feat),self._gram_matrix(teacher_feat))

        # 边缘损失
        # a = self._sobel(student_feat)
        # b = self._sobel(teacher_feat)
        # print('self._sobel(student_feat)',a.shape)
        edge_loss = F.mse_loss(self._sobel(student_feat),self._sobel(teacher_feat))

        return style_loss + edge_loss

    def _gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        # print('features',features.shape)
        gram = torch.bmm(features, features.transpose(1, 2))
        # print('gram',gram.shape)
        return gram / (c * h * w)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        gap = self.gap(x).view(b, c)
        gmp = self.gmp(x).view(b, c)
        attn = self.fc(gap + gmp).view(b, c, 1, 1)
        return x * attn.expand_as(x)



if __name__ == '__main__':
    # diffuser = TextureDiffuser(input=64, latent_dim=24).cuda()
    diffuser = ProgressiveDiffusion(in_channels=32).cuda()
    rgb = torch.randn([2, 32, 120, 160]).cuda()
    d = torch.randn([2, 32, 120, 160]).cuda()
    hf_loss = diffuser(rgb, d)
    hfloss = hf_loss.item()
    print("==> Total params: %.2fM" % (sum(p.numel() for p in diffuser.parameters()) / 1e6))
    # print("s.shape:", s[0][-1].shape)
    # print("s.shape:", s[1][-1].shape)
    # print("s.shape:", s[2][-1].shape)
    # print("s.shape:", s[].shape)
    print("hf_loss:", hfloss)
    # print("s.shape:", s[4][-1].shape)
