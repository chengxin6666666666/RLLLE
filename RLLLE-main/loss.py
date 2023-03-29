from daytime_inference.models.pix2pix_model import Pix2PixModel
import numpy as np
import torch
import torch.nn as nn
from guided_filter_pytorch.guided_filter import GuidedFilter
import os
from Net import *
import torchvision
import copy





#下面四段是计算 vgg loss

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h

        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        relu4_1 = h
        h = F.relu(self.conv4_2(h), inplace=True)
        relu4_2 = h
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)
        relu4_3 = h


        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        relu5_2 = F.relu(self.conv5_2(relu5_1), inplace=True)
        conv5_3 = self.conv5_3(relu5_2)
        h = F.relu(conv5_3, inplace=True)
        relu5_3 = h

        return relu5_1



def vgg_preprocess(batch,gray=False):
    tensortype = type(batch.data)
    if gray==True:
        batch = batch.repeat(1, 3, 1, 1)
        batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
        return batch
    else:
        (r, g, b) = torch.chunk(batch, 3, dim=1)
        batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
        batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
        return batch


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute_vgg_loss(self, vgg, img, target,gray):
        img_vgg = vgg_preprocess(img,gray)
        target_vgg = vgg_preprocess(target,gray)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)


        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
    #     if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
    #         os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
    #     vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
    #     vgg = Vgg16()
    #     for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
    #         dst.data[:] = src
    #     torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    # vgg.cuda()
    vgg.cuda()
    vgg.load_state_dict(torch.load( './model/vgg16.weight'),strict=False)
    # vgg = torch.nn.DataParallel(vgg, gpu_ids)
    return vgg









#MIRNet-v2的loss
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss




#zero-dce的loss
class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg,2)
        Drb = torch.pow(mr - mb,2)
        Dgb = torch.pow(mb - mg,2)
        k =torch.pow(torch.pow( Drg ,2)+ torch.pow( Drb ,2) +torch.pow( Dgb ,2),0.5)

        # Drg =torch.nn.L1Loss(mr-mg)
        # Drb = torch.nn.L1Loss(mr-mb)
        # Dgb = torch.nn.L1Loss(mb-mg)
        # k = Drg+ Drb +Dgb

        return k




#
# class L_Grad(nn.Module):
#     def __init__(self):
#         super(L_Grad, self).__init__()
#         self.sobelconv=Sobelxy()
#
#     def forward(self, image_A, image_B):
#         image_A_lf,image_A_hf=get_LFHF(image_A)
#         image_B_lf, image_B_hf = get_LFHF(image_B)
#         image_A_Y = image_A_lf[:, :1, :, :]
#         image_B_Y = image_B_lf[:, :1, :, :]
#         image_A_Y2 = image_A_lf[:, 1:2, :, :]
#         image_B_Y2 = image_B_lf[:, 1:2, :, :]
#         image_A_Y3 = image_A_lf[:, 2:3, :, :]
#         image_B_Y3 = image_B_lf[:, 2:3, :, :]
#
#         gradient_A = self.sobelconv(image_A_Y)
#         gradient_B = self.sobelconv(image_B_Y)
#
#         gradient_A2 = self.sobelconv(image_A_Y2)
#         gradient_B2 = self.sobelconv(image_B_Y2)
#
#         gradient_A3 = self.sobelconv(image_A_Y3)
#         gradient_B3 = self.sobelconv(image_B_Y3)
#
#
#
#         Loss_gradient1 = F.l1_loss(gradient_A,gradient_B)
#         Loss_gradient2 = F.l1_loss(gradient_A2, gradient_B2)
#         Loss_gradient3 = F.l1_loss(gradient_A3,gradient_B3)
#
#         Loss_gradient=(Loss_gradient1+Loss_gradient2+Loss_gradient3)/3
#
#
#
#         return Loss_gradient
#
# class Sobelxy(nn.Module):
#     def __init__(self):
#         super(Sobelxy, self).__init__()
#         kernelx = [[-1, 0, 1],
#                   [-2,0 , 2],
#                   [-1, 0, 1]]
#         kernely = [[1, 2, 1],
#                   [0,0 , 0],
#                   [-1, -2, -1]]
#         kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
#         kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
#         self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
#         self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
#     def forward(self,x):
#         sobelx=F.conv2d(x, self.weightx, padding=1)
#         sobely=F.conv2d(x, self.weighty, padding=1)
#         return torch.abs(sobelx)+torch.abs(sobely)








