import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import numpy as np
import cv2

#region以外snr map是什么意思我没明白，我是在174行对应是1-SNR map

from models.archs.transformer.Models import Encoder_patch66

###############################
class low_light_transformer(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(low_light_transformer, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)

        self.upconv1 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf*2, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64*2, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer_ns = Encoder_patch66(d_model=1024, d_inner=2048, n_layers=6)                   #极低的transformer
        self.transformer_glow = Encoder_patch66(d_model=1024, d_inner=2048, n_layers=6)                 #过曝光的transformer
        self.transformer_lol = Encoder_patch66(d_model=1024, d_inner=2048, n_layers=6)                  #均匀的transformer
        self.recon_trunk_light_lol = arch_util.make_layer(ResidualBlock_noBN_f, 6)                      #均匀的CNN
        self.recon_trunk_light_ns = arch_util.make_layer(ResidualBlock_noBN_f, 6)                       #极低的CNN
        self.recon_trunk_light_glow = arch_util.make_layer(ResidualBlock_noBN_f, 6)                     #过曝光的CNN

    def forward(self, x, mask=None,map=None):
        x_center = x



        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))

        fea = self.feature_extraction(L1_fea_3)

        #通过分别是均匀、极低、过曝光三者的CNN编码
        fea_light_lol = self.recon_trunk_light_lol(fea)
        fea_light_ns = self.recon_trunk_light_ns(fea)
        fea_light_glow = self.recon_trunk_light_glow(fea)


        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        channel = fea.shape[1]
        #得到三个区的mask
        mask_lol=mask[:, 1, :, :]
        mask_lol=torch.unsqueeze(mask_lol,dim=1)
        mask_ns = mask[:, 0, :, :]
        mask_ns=torch.unsqueeze(mask_ns, dim=1)
        mask_glow = mask[:, 2, :, :]
        mask_glow=torch.unsqueeze(mask_glow, dim=1)


        mask2=mask
        trans_mask_lol = (mask2 == mask2.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)

        trans_mask_lol_lol = trans_mask_lol[:, 1, :, :].unsqueeze(dim=1)

        trans_mask_lol_ns = (trans_mask_lol[:, 1, :, :]+trans_mask_lol[:, 0, :, :]).unsqueeze(dim=1)

        trans_mask_lol_glow = (trans_mask_lol[:, 1, :, :] + trans_mask_lol[:, 2, :, :]).unsqueeze(dim=1)

        mask_map_lol=trans_mask_lol_lol*map

        mask_map_ns = trans_mask_lol_ns * map

        mask_map_glow = trans_mask_lol_glow * map








        #resize
        mask_lol = F.interpolate(mask_lol, size=[h_feature, w_feature], mode='nearest')


        mask_ns = F.interpolate(mask_ns, size=[h_feature, w_feature], mode='nearest')

        mask_glow = F.interpolate(mask_glow, size=[h_feature, w_feature], mode='nearest')

        mask_map_lol = F.interpolate(mask_map_lol, size=[h_feature, w_feature], mode='nearest')

        mask_map_glow = F.interpolate(mask_map_glow, size=[h_feature, w_feature], mode='nearest')

        mask_map_ns = F.interpolate(mask_map_ns, size=[h_feature, w_feature], mode='nearest')







        #极低
        xs = np.linspace(-1, 1, fea.size(3) // 4)
        ys = np.linspace(-1, 1, fea.size(2) // 4)
        xs = np.meshgrid(xs, ys)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(fea.size(0), 1, 1, 1).cuda()
        xs = xs.view(fea.size(0), -1, 2)


        height = fea.shape[2]
        width = fea.shape[3]
        fea_unfold = F.unfold(fea, kernel_size=4, dilation=1, stride=4, padding=0)
        fea_unfold = fea_unfold.permute(0, 2, 1)

        mask_unfold_ns = F.unfold(mask_map_ns, kernel_size=4, dilation=1, stride=4, padding=0)
        mask_unfold_ns = mask_unfold_ns.permute(0, 2, 1)
        mask_unfold_ns = torch.mean(mask_unfold_ns, dim=2).unsqueeze(dim=-2)
        mask_unfold_ns[mask_unfold_ns <= 0.5] = 0.0                                             #极低部分：过曝光的SNR map比均匀大，要去除过曝光就把高的赋0

        fea_unfold_ns = self.transformer_ns(fea_unfold, xs, src_mask=mask_unfold_ns)
        fea_unfold_ns = fea_unfold_ns.permute(0, 2, 1)
        fea_unfold_ns = nn.Fold(output_size=(height, width), kernel_size=(4, 4), stride=4, padding=0, dilation=1)(fea_unfold_ns)

        channel = fea.shape[1]





        #过曝光
        xs = np.linspace(-1, 1, fea.size(3) // 4)
        ys = np.linspace(-1, 1, fea.size(2) // 4)
        xs = np.meshgrid(xs, ys)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(fea.size(0), 1, 1, 1).cuda()
        xs = xs.view(fea.size(0), -1, 2)

        height = fea.shape[2]
        width = fea.shape[3]
        fea_unfold = F.unfold(fea, kernel_size=4, dilation=1, stride=4, padding=0)
        fea_unfold = fea_unfold.permute(0, 2, 1)

        mask_unfold_glow = F.unfold(mask_map_glow, kernel_size=4, dilation=1, stride=4, padding=0)
        mask_unfold_glow = mask_unfold_glow.permute(0, 2, 1)
        mask_unfold_glow = torch.mean(mask_unfold_glow, dim=2).unsqueeze(dim=-2)
        mask_unfold_glow[mask_unfold_glow >= 0.8] = 0.0                          #过曝光部分：极低的SNR map比均匀小，要去除极低部分，把低的赋0

        fea_unfold_glow = self.transformer_glow(fea_unfold, xs, src_mask=mask_unfold_glow)
        fea_unfold_glow = fea_unfold_glow.permute(0, 2, 1)
        fea_unfold_glow = nn.Fold(output_size=(height, width), kernel_size=(4, 4), stride=4, padding=0, dilation=1)(
            fea_unfold_glow)











        #均匀
        mask_unfold_lol = F.unfold(mask_map_lol, kernel_size=4, dilation=1, stride=4, padding=0)
        mask_unfold_lol = mask_unfold_lol.permute(0, 2, 1)
        mask_unfold_lol = torch.mean(mask_unfold_lol, dim=2).unsqueeze(dim=-2)
        mask_unfold_lol[mask_unfold_lol <= 0.5] = 0.0

        fea_unfold_lol = self.transformer_lol(fea_unfold, xs, src_mask=mask_unfold_lol)
        fea_unfold_lol = fea_unfold_lol.permute(0, 2, 1)
        fea_unfold_lol = nn.Fold(output_size=(height, width), kernel_size=(4, 4), stride=4, padding=0, dilation=1)(fea_unfold_lol)

        channel = fea.shape[1]




        mask_lol = mask_lol.repeat(1, channel, 1, 1)


        mask_ns = mask_ns.repeat(1, channel, 1, 1)

        mask_glow = mask_glow.repeat(1, channel, 1, 1)

        mask_map_lol = mask_map_lol.repeat(1, channel, 1, 1)

        mask_map_ns = mask_map_ns.repeat(1, channel, 1, 1)

        mask_map_glow = mask_map_glow.repeat(1, channel, 1, 1)

        #先在内部乘上SNR map融合，在外部乘上分割mask融合
        fea = ((fea_light_ns*mask_map_ns)+(1-mask_map_ns)*fea_unfold_ns) * mask_ns + \
              ((fea_light_lol*mask_map_lol)+(1-mask_map_lol)*fea_unfold_lol) * mask_lol  +\
              ((fea_light_glow*mask_map_glow)+(1-mask_map_glow)*fea_unfold_glow)*mask_glow

        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise = self.conv_last(out_noise)
        out_noise = out_noise + x_center


        return out_noise