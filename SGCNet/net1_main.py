import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from mymodelnew.toolbox.Mymodels.Baseline_lsy_seg.part import AutoSelectedSparseAttention
from mymodelnew.toolbox.Mymodels.Baseline_lsy_seg.part import ChannelOrthogonalFilterAttention
from mymodelnew.toolbox.Mymodels.Baseline_lsy_seg.MLPDecoder import DecoderHead
# from mymodelnew.Backbone.P2T.p2t import p2t_base
from mymodelnew.Backbone.P2T.p2t import p2t_small


class SACAM(nn.Module):
    def __init__(self, ch_in, h, w, reduction=16):
        super(SACAM, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
        self.fc_wight = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, 1, bias=False),
            nn.Sigmoid()
        )

        self.assa = AutoSelectedSparseAttention(ch_in)
        self.OCA = ChannelOrthogonalFilterAttention(ch_in, h, w)


    def forward(self, x_h, x_l):
        b, c, _, _ = x_h.size()
        y_h = x_h.mean(dim=(2,3)) # squeeze操作
        h_weight = self.fc_wight(y_h)
        y_h = self.fc(y_h).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        x_fusion_h = x_h * y_h.expand_as(x_h)
        x_fusion_h = torch.mul(x_fusion_h, h_weight.view(b, 1, 1, 1))
        x_fusion_h = self.assa(x_fusion_h)
##################----------------------------------
        b, c, _, _ = x_l.size()
        y_l = x_l.mean(dim=(2,3)) # squeeze操作
        l_weight = self.fc_wight(y_l)
        y_l = self.fc(y_l).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        x_fusion_l = x_l * y_l.expand_as(x_l)
        x_fusion_l = torch.mul(x_fusion_l, l_weight.view(b, 1, 1, 1))
        x_fusion_l = self.cofa(x_fusion_l)
#################-------------------------------
        x_fusion = x_fusion_h+x_fusion_l

        return x_fusion  # 注意力作用每一个通道上

class Net1(nn.Module):
    def __init__(self,num_class=41,):
        super(Net1,self).__init__()
        # self.backbone = p2t_base()
        self.backbone = p2t_small()


        self.Mlp_decoder = DecoderHead()
        dim = [64,128,160,256]


        self.conv320to160 = nn.Conv2d(in_channels=320, out_channels=160, kernel_size=1)
        self.conv512to256 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)


        self.sig = nn.Sigmoid()


        self.SAM1 = SACAM(dim[0],120,160)
        self.SAM2 = SACAM(dim[1],60,80)
        self.SAM3 = SACAM(dim[2],30,40)
        self.SAM4 = SACAM(dim[3],15,20)


        self.pipei = nn.Conv2d(64,256,1)



    def forward(self, rgb, dep):
        rgb = self.backbone(rgb)
        depth = self.backbone(dep)
        rgb[2] = self.conv320to160(rgb[2])
        depth[2] = self.conv320to160(depth[2])
        rgb[3] = self.conv512to256(rgb[3])
        depth[3] = self.conv512to256(depth[3])

        x1 = self.sig(rgb[0])
        x2 = self.sig(rgb[1])
        x3 = self.sig(rgb[2])
        x4 = self.sig(rgb[3])

        fuse0 = self.SAM1(rgb[0], depth[0])
        fuse0 = (x1 * fuse0) + fuse0
        fuse1 = self.SAM2(rgb[1], depth[1])
        fuse1 = (x2 * fuse1) + fuse1
        fuse2 = self.SAM3(rgb[2], depth[2])
        fuse2 = (x3 * fuse2) + fuse2
        fuse3 = self.SAM4(rgb[3], depth[3])
        fuse3 = (x4 * fuse3) + fuse3




        out = self.Mlp_decoder(fuse0, fuse1, fuse2, fuse3)

        ##mutal
        fuse0 = self.pipei(fuse0)
        fuse3 = F.interpolate(fuse3, size=fuse0.shape[2:], mode='bilinear', align_corners=False)


        return out,fuse0,fuse1,fuse2,fuse3

    def load_pre_sa(self, pre_model):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model)['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        self.backbone.load_state_dict(new_state_dict3, strict=False)
        print('self.backbone_seg_mit loading')


if __name__ == '__main__':
    net = Net1().cuda()
    rgb = torch.randn([1, 3, 480, 640]).cuda()
    d = torch.randn([1, 3, 480, 640]).cuda()
    s = net(rgb, d)
    print("==> Total params: %.2fM" % (sum(p.numel() for p in net.parameters()) / 1e6))
    from mymodelnew.toolbox.Mymodels.Baseline_lsy_seg.FLOP import CalParams
    CalParams(net, rgb, d)
    # print("s.shape:", s[0][-1].shape)
    # print("s.shape:", s[1][-1].shape)
    # print("s.shape:", s[1][-1].shape)
    # print("s.shape:", s[4][-1].shape)

    import torch
    from thop import profile

    # 假设你的模型
    model = MyNet()
    input = torch.randn(1, 3, 224, 224)  # 单模态输入，比如 RGB
    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
    from ptflops import get_model_complexity_info


    model = MyNet()
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,print_per_layer_stat=True)
        print('FLOPs:', macs)
        print('Params:', params)


    from torchsummary import summary
    model = MyNet()
    summary(model, input_size=(3, 224, 224))  # 单模态输入