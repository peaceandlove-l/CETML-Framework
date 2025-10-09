import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDLoss(nn.Module):
    def __init__(self, alpha=1, tau=1, resize_config=None, shuffle_config=None, transform_config=None,
                 warmup_config=None, earlydecay_config=None):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau

        self.resize_config = resize_config
        self.shuffle_config = shuffle_config
        self.transform_config = transform_config
        self.warmup_config = warmup_config
        self.earlydecay_config = earlydecay_config

        self.KLD = torch.nn.KLDivLoss(reduction='sum')


    def forward(self, x_student, x_teacher):
        x_student = F.log_softmax(x_student / self.tau, dim=-1)
        x_teacher = F.softmax(x_teacher / self.tau, dim=-1)
        loss = self.KLD(x_student, x_teacher) / (x_student.numel() / x_student.shape[-1])
        loss = self.alpha * loss
        return loss


import scipy.fftpack as fftpack
# -------------------------- 频域变换 --------------------------
class DCTLayer(nn.Module):
    """离散余弦变换(DCT)层"""
    def __init__(self, norm='ortho'):
        super().__init__()
        self.norm = norm

    def forward(self, x):                        # x: [B, C, H, W]
        x_np = x.cpu().numpy()
        x_np = fftpack.dct(x_np, axis=2, norm=self.norm)  # H 维
        x_np = fftpack.dct(x_np, axis=3, norm=self.norm)  # W 维
        return torch.from_numpy(x_np).to(x.dtype).to(x.device)


class IDCTLayer(nn.Module):
    """逆离散余弦变换(IDCT)层"""
    def __init__(self, norm='ortho'):
        super().__init__()
        self.norm = norm

    def forward(self, x):                        # x: [B, C, H, W]
        x_np = x.cpu().numpy()
        x_np = fftpack.idct(x_np, axis=2, norm=self.norm)
        x_np = fftpack.idct(x_np, axis=3, norm=self.norm)
        return torch.from_numpy(x_np).to(x.dtype).to(x.device)


# -------------------------- 边缘一致性 --------------------------
class EdgeConsistencyLoss(nn.Module):
    """边缘一致性损失"""
    def __init__(self, edge_threshold=0.3, eps=1e-5):
        super().__init__()
        self.edge_threshold = edge_threshold
        self.eps = eps

        # Sobel 滤波器（固定权重）
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self._init_sobel()
        for p in self.parameters():
            p.requires_grad = False

    def _init_sobel(self):
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x.weight.data.copy_(kx)
        self.sobel_y.weight.data.copy_(ky)

    def _edge_map(self, x):
        gray = x.mean(1, keepdim=True) if x.shape[1] > 1 else x
        gx, gy = self.sobel_x(gray), self.sobel_y(gray)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + self.eps)
        denom = (mag.max() - mag.min()).clamp(min=self.eps)
        return (mag - mag.min()) / denom          # 归一化至 [0,1]

    def forward(self, pred1, pred2, edge_map):
        e1, e2 = self._edge_map(pred1), self._edge_map(pred2)
        mask = (edge_map > self.edge_threshold).float()
        return F.mse_loss(e1 * mask, e2 * mask)


# ----------------------- EnhancedMutualLoss -------------------
class EnhancedMutualLoss(nn.Module):
    """
    KL + 特征重建 + 边缘一致性
    """
    def __init__(self, alpha=1.0, beta=0.7, gamma=0.3, tau=1.0,
                 resize_config=None, edge_aware=True, freq_domain=False,
                 dynamic_weighting=True, warmup_epochs=10):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.tau = tau

        self.edge_aware = edge_aware
        self.freq_domain = freq_domain
        self.dynamic_weighting = dynamic_weighting
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        self.adap_pool = nn.AdaptiveAvgPool2d(resize_config) if resize_config else None
        self.kld = nn.KLDivLoss(reduction='batchmean')
        self.edge_loss = EdgeConsistencyLoss() if edge_aware else None

        if freq_domain:
            self.dct = DCTLayer()
            self.idct = IDCTLayer()
        else:
            self.dct = self.idct = None

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    # 动态权重
    def _dyn_w(self):
        warm = min(1.0, self.current_epoch / self.warmup_epochs)
        edge_w = self.gamma * warm
        if self.current_epoch > self.warmup_epochs:
            feat_w = self.beta * (0.95 ** (self.current_epoch - self.warmup_epochs))
        else:
            feat_w = self.beta
        return edge_w, feat_w

    def forward(self, x_s, x_t, edge_map, feat_s=None, feat_t=None):

        # -------- KL 散度 --------
        log_s = F.log_softmax(x_s / self.tau, 1)
        soft_t = F.softmax(x_t / self.tau, 1)
        kld_loss = self.kld(log_s, soft_t)
        total = self.alpha * kld_loss

        # edge_c_loss = x_s.new_zeros(())
        # feat_rec_loss = x_s.new_zeros(())

        edge_w, feat_w = self._dyn_w() if self.dynamic_weighting else (self.gamma, self.beta)

        # -------- 特征重建 --------
        if feat_s is not None and feat_t is not None:
            # print('1')
            losses = []
            for fs, ft in zip(feat_s, feat_t):
                if self.adap_pool:
                    fs, ft = self.adap_pool(fs), self.adap_pool(ft)

                if self.freq_domain and self.dct:
                    fs_f, ft_f = self.dct(fs.unsqueeze(0)), self.dct(ft.unsqueeze(0))
                    freq_l = F.l1_loss(fs_f[:, :, 1:, 1:], ft_f[:, :, 1:, 1:])
                    fs_sp, ft_sp = self.idct(fs_f), self.idct(ft_f)
                    spa_l = 1 - F.cosine_similarity(fs_sp.flatten(1), ft_sp.flatten(1)).mean()
                    losses.append(freq_l + spa_l)
                else:
                    losses.append(1 - F.cosine_similarity(fs.flatten(1), ft.flatten(1)).mean())
            feat_rec_loss = sum(losses) / len(losses)
            total = total + feat_w * feat_rec_loss

        return total



if __name__ == '__main__':
    # recon = ReconKLDLoss(tau=1).cuda()
    recon = EnhancedMutualLoss(freq_domain=True).cuda()
    rgb = torch.randn([2, 41, 120, 160]).cuda()
    d = torch.randn([2, 41, 120, 160]).cuda()
    loss = recon(rgb,d,rgb,rgb,d)
    # print("==> Total params: %.2fM" % (sum(p.numel() for p in recon.parameters()) / 1e6))
    print("total loss:", loss)
    # print("s.shape:", s[4][-1].shape)