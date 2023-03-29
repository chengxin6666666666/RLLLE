import logging
from collections import OrderedDict
import loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import GuidedFilter
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, CharbonnierLoss2
from daytime_inference.models.pix2pix_model import Pix2PixModel
import time
import  copy
from  image_pool import ImagePool
import random
logger = logging.getLogger('base')
import  net

#极低和过曝光的vgg部分我去掉列，我先跑了1k，那个合成图vgg会过曝光部分变得很割裂


#22eccv的fuse gray
def rgb2gray(rgb):
    rgb2=rgb

    # gray = 0.2989 * rgb[:,  0:1, :, :] + 0.5870 * rgb[:, 1:2, :, :] + 0.1140 * rgb[:, 2:3,: , :]


    gray_r=torch.exp((-1*torch.pow((rgb2[:,  0:1, :, :]-0.5),2))/0.08)
    gray_g = torch.exp((-1 * torch.pow((rgb2[:, 1:2, :, :] - 0.5), 2)) / 0.08)
    gray_b = torch.exp((-1 * torch.pow((rgb2[:, 2:3, :, :] - 0.5), 2)) / 0.08)
    gray=(gray_r*rgb[:,  0:1, :, :]+gray_g*rgb[:,  1:2, :, :]+gray_b*rgb[:,  2:3, :, :])/3

    # rgb[:, 0:1, :, :]=gray_r * rgb[:, 0:1, :, :]
    # rgb[:, 1:2, :, :]  = gray_g * rgb[:, 1:2, :, :]
    # rgb[:, 2:3, :, :] =gray_b * rgb[:, 2:3, :, :]


    return gray



#原本的rgb转gray
def orirgb2gray(rgb):
    rgb2=rgb

    gray = 0.2989 * rgb[:,  0:1, :, :] + 0.5870 * rgb[:, 1:2, :, :] + 0.1140 * rgb[:, 2:3,: , :]




    return gray


#导向滤波器分离高频低频分量
def get_LFHF(image, rad_list=[2], eps_list=[1e-8]):
    def decomposition(guide, inp, rad_list, eps_list):
        LF_list = []
        HF_list = []
        for radius in rad_list:
            for eps in eps_list:
                gf = GuidedFilter(radius, eps)
                LF = gf(guide, inp)
                LF[LF > 1] = 1
                LF_list.append(LF)
                HF_list.append(inp - LF)
        LF = torch.cat(LF_list, dim=1)
        HF = torch.cat(HF_list, dim=1)
        return LF, HF

    image = torch.clamp(image, min=0.0, max=1.0)
    # Compute the LF-HF features of the image
    img_lf, img_hf = decomposition(guide=image,
                                   inp=image,
                                   rad_list=rad_list,
                                   eps_list=eps_list)
    return img_lf, img_hf

class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.Tensor = torch.cuda.FloatTensor
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # self.netD=net.define_D(3, 64,'no_norm_4',5, 'instance', True, 0, False)
        # self.netD_P = net.define_D(3, 64,'no_norm_4',4, 'instance', True, 0, True)
        # self.optimizer_D_A = torch.optim.Adam(self.netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
        #
        # self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=2e-4, betas=(0.5, 0.999))
        # print network
        self.print_network()
        self.load()


        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'cb2':
                self.cri_pix = CharbonnierLoss2().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            self.cri_pix_ill = nn.L1Loss(reduction='sum').to(self.device)
            self.CharbonnierLoss=loss.CharbonnierLoss()
            self.L_color=loss.L_color()
            # self.grad=loss.L_Grad()
            self.vgg_loss = loss.PerceptualLoss()
            self.vgg_loss.cuda()
            self.vgg = loss.load_vgg16("./model")
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
            # self.fake_H_pool = ImagePool(50)
            # self.criterionGAN = net.GANLoss(use_lsgan=False,tensor=self.Tensor)


            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

    # def backward_D_basic(self, netD, real, fake, use_ragan):
    #     # Real
    #     pred_real = netD.forward(real)
    #     pred_fake = netD.forward(fake.detach())
    #
    #     loss_D_real = 10*self.criterionGAN(pred_real, True)
    #     loss_D_fake = 10*self.criterionGAN(pred_fake, False)
    #     loss_D = (loss_D_real + loss_D_fake) * 0.5
    #     # loss_D.backward()
    #     return loss_D
    #
    # def backward_D_A(self):
    #     fake_B = self.fake_H_pool.query(self.fake_H)
    #     fake_B = self.fake_H
    #     self.loss_D_A = self.backward_D_basic(self.netD, self.DAY, fake_B, True)
    #     self.loss_D_A.backward()
    #
    # def backward_D_P(self):
    #
    #     loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, False)
    #
    #     for i in range(5):
    #         loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], False)
    #         self.loss_D_P = loss_D_P / float(5+ 1)
    #
    #
    #
    #
    #     self.loss_D_P.backward()



    #拼接图VGG Loss
    def fused_loss(self,x,y):
        # x=orirgb2gray(x)
        loss_vgg = self.vgg_loss.compute_vgg_loss(self.vgg, x, y, False)
        return loss_vgg



    #极低部分的loss
    def extremelow_loss(self,x, y):
        l1 = nn.L1Loss()
        x_lf, x_hf = get_LFHF(x)
        y_lf, y_hf = get_LFHF(y)

        loss_l1 = l1(x_lf, y_lf)

        # loss_vgg = self.vgg_loss.compute_vgg_loss(self.vgg,x_lf, y_lf,False)
        loss =  loss_l1
        return loss


    #过曝光部分的loss
    def light_loss(self,x, y):
        l1 = nn.L1Loss()
        # x_gray = orirgb2gray(x)    #原始gray
        # y_gray = rgb2gray(y)        #fuse gray
        # x_gray_lf, x_gray_hf = get_LFHF(x)
        # y_gray_lf, y_gray_hf = get_LFHF(y)

        loss_l1 = l1(x, y)





        # loss_l1 = l1(x_gray_lf, y_gray_lf)
        # loss_vgg = self.vgg_loss.compute_vgg_loss(self.vgg,x_gray_hf, y_gray_hf,False)

        # x_lf, x_hf = get_LFHF(x)
        # y_lf, y_hf = get_LFHF(y)
        # loss_l1 = l1(x_hf, y_hf)
        # loss_vgg = self.vgg_loss.compute_vgg_loss(self.vgg, x, y,False)
        # loss =loss_l1+loss_vgg
        return loss

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        self.Mask = data['MASK'].to(self.device)
        self.vsait_Mask = data['DAY'].to(self.device)                                   #NS的非0区mask
        self.var_L_hist = data['img_LQ_HIST'].to(self.device)
        self.nf = data['nf'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)


    #task loss
    def get_task_loss(self, fake_img, step):
        if step < 2:
            task_model = Pix2PixModel()
            load_filename = 'latest_net_G.pth'
            load_path = './daytime_inference/checkpoints/'
            load_file = load_path + load_filename
            self.task_net = getattr(task_model, 'netG')  # net = model.netG_A / netG_B
            self.task_net.eval()
            if isinstance(self.task_net, torch.nn.DataParallel):
                self.task_net = self.task_net.module
            if torch.cuda.is_available():
                _device = 'cuda:0'
            else:
                _device = 'cpu'
            _state_dict = torch.load(load_file, map_location=str(_device))
            self.task_net.load_state_dict(_state_dict,strict=False)
            # print('\nloading the model paras from {} on device--{}\n'.format(load_file, _device))
        daytime_tensor_fake_B = torch.mean(self.task_net(fake_img), dim=0)
        return (-1 * daytime_tensor_fake_B + 1).mean() / 2.

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        # self.optimizer_G.zero_grad()
        #计算SNR MAP

        dark = self.var_L
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = self.nf
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)

        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max+0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()




        #前向传播
        self.fake_H = self.netG(self.var_L, self.Mask,mask)
        # self.opt.patchSize = 32
        #
        # w = self.var_L.size(3)
        # h = self.var_L.size(2)
        # w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
        # h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))
        #
        #
        # self.fake_patch = self.fake_H[:, :, h_offset:h_offset + self.opt.patchSize,
        #                   w_offset:w_offset + self.opt.patchSize]
        # self.real_patch = self.DAY[:, :, h_offset:h_offset + self.opt.patchSize,
        #                   w_offset:w_offset + self.opt.patchSize]
        # self.input_patch = self.var_L[:, :, h_offset:h_offset + self.opt.patchSize,
        #                    w_offset:w_offset + self.opt.patchSize]
        #
        # self.fake_patch_1 = []
        # self.real_patch_1 = []
        # self.input_patch_1 = []
        # w = self.var_L.size(3)
        # h = self.var_L.size(2)
        # for i in range(5):
        #     w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
        #     h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
        #     self.fake_patch_1.append(self.fake_H[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
        #                              w_offset_1:w_offset_1 + self.opt.patchSize])
        #     self.real_patch_1.append(self.DAY[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
        #                              w_offset_1:w_offset_1 + self.opt.patchSize])
        #     self.input_patch_1.append(self.var_L[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
        #                               w_offset_1:w_offset_1 + self.opt.patchSize])


        #取出各个区的mask
        mask2=self.Mask

        mask_lol = (mask2 == mask2.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)


        mask_lol_lol=(mask_lol[:,1,:,:]+mask_lol[:,2,:,:]).unsqueeze(dim=1)

        mask_lol_lol=mask_lol_lol.repeat(1,3,1,1)

        mask_lol_lol2 =( mask_lol[:, 2, :, :]).unsqueeze(dim=1)

        mask_lol_lol2 = mask_lol_lol2.repeat(1, 3, 1, 1)



        mask_lol_ns = (mask_lol[:,0,:,:]).unsqueeze(dim=1)                      #极低

        mask_lol_ns = mask_lol_ns.repeat(1, 3, 1, 1)

        mask_lol_glow =( mask_lol[:, 2, :, :]).unsqueeze(dim=1)                     #过曝光

        mask_lol_glow = mask_lol_glow.repeat(1, 3, 1, 1)

        mask_lol_exl =( mask_lol[:, 0, :, :]).unsqueeze(dim=1)

        mask_lol_exl = mask_lol_exl.repeat(1, 3, 1, 1)





                                                                                        #均匀区
        self.fake_H_lol = self.fake_H * mask_lol_lol
        self.real_H_lol = self.real_H * mask_lol_lol
        l_Cr=self.CharbonnierLoss(self.fake_H_lol,self.real_H_lol)                      #MIRNet-v2 loss


        self.fake_H_exl=self.fake_H*mask_lol_exl
        self.var_L_hist_exl=self.var_L_hist*mask_lol_exl                                #极低区
        l_exl=self.extremelow_loss(self.fake_H_exl,self.var_L_hist_exl)                 #计算极低loss


        l_color=torch.mean(self.L_color(self.fake_H))

        #
        self.fake_H_glow = self.fake_H*mask_lol_glow
        self.vsait_glow =self.vsait_Mask*mask_lol_glow                                #过曝光区
        # # l_glow=self.light_loss(self.fake_H_glow,self.vsait_glow)                        #计算过曝光loss
        # l_glow=self.CharbonnierLoss(self.fake_H_glow,self.vsait_glow)



        l_task=self.get_task_loss(fake_img=self.fake_H,step=step)                        #task loss


        self.fake_H_lol2 = self.fake_H * mask_lol_lol
        self.real_H_lol2 = self.real_H * mask_lol_lol


        # fused_glow=fused_gray.repeat(1, 3, 1, 1)                              #过曝光取fused gray并且通道×3
        # self.fused=orirgb2gray(self.real_H_lol)+orirgb2gray(self.var_L_hist_exl)+fused_gray                  #生成拼接图
        self.fused = self.real_H_lol + self.var_L_hist_exl + self.vsait_glow
        l_vgg=self.fused_loss(self.fake_H,self.fused)
        # pred_fake = self.netD.forward(self.fake_H)
        # pred_real = self.netD.forward(self.DAY)
        #
        # self.loss_G_A = (self.criterionGAN(pred_real - torch.mean(pred_fake), False) +
        #                  self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
        #
        # pred_fake_patch = self.netD_P.forward(self.fake_patch)
        # loss_G_A = 0
        #
        # loss_G_A += self.criterionGAN(pred_fake_patch, True)
        #
        #
        # for i in range(5):
        #     pred_fake_patch_1 = self.netD_P.forward(self.fake_patch_1[i])
        #
        #     loss_G_A += self.criterionGAN(pred_fake_patch_1, True)
        #
        #     self.loss_G_A += loss_G_A / float(5+ 1) * 2


        l_final = 0.1 * l_task +   0.5 * l_color + 1 * l_exl+1*l_Cr+0.1*0.1*l_vgg
        # l_final = 0.1 * l_task +0.1 * l_exl + 1 * l_Cr + 1 * l_glow
        self.optimizer_G.zero_grad()
        l_final.backward()
        self.optimizer_G.step()
        # self.optimizer_D_A.zero_grad()
        # self.backward_D_A()
        # self.optimizer_D_P.zero_grad()
        # self.backward_D_P()
        # self.optimizer_D_A.step()
        # self.optimizer_D_P.step()
        #
        # loss_l_Cr=l_Cr.item()
        loss_l_task = l_task.item()
        # loss_l_grad =  l_grad.item()
        # loss_l_glow = l_glow.item()
        # loss_l_exl = l_exl.item()
        loss_l_color = l_color.item()
        self.log_dict = OrderedDict(
            [('l_task', loss_l_task),
             ('l_color', loss_l_color)])

        # self.log_dict=OrderedDict([('l_cr',loss_l_Cr),('l_task',loss_l_task),('l_grad',loss_l_grad),('l_glow',loss_l_glow),('l_exl',loss_l_exl),('l_color',loss_l_color)])


    def test(self):
        self.netG.eval()
        with torch.no_grad():
            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max+0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()
            self.fake_H = self.netG(self.var_L, self.Mask,mask)
        self.netG.train()

    def test4(self):
        self.netG.eval()
        with torch.no_grad():

            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max + 0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()

            # var_L = self.var_L.clone().view(B, C, H, W)
            # H_new = 400
            # W_new = 608
            # var_L = F.interpolate(var_L, size=[H_new, W_new], mode='bilinear')
            # mask = F.interpolate(mask, size=[H_new, W_new], mode='bilinear')
            # var_L = var_L.view(B, C, H_new, W_new)

            self.fake_H = self.netG(self.var_L, self.Mask,mask)
            # self.fake_H = F.interpolate(self.fake_H, size=[H, W], mode='bilinear')


            torch.cuda.empty_cache()

        self.netG.train()


    def test5(self):
        self.netG.eval()
        self.fake_H = None
        with torch.no_grad():
            B, C, H, W = self.var_L.size()

            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max + 0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()

            del light
            del dark
            del noise
            torch.cuda.empty_cache()

            var_L = self.var_L.clone().view(B, C, H, W)
            H_new = 384
            W_new = 384
            var_L = F.interpolate(var_L, size=[H_new, W_new], mode='bilinear')
            mask = F.interpolate(mask, size=[H_new, W_new], mode='bilinear')
            var_L = var_L.view(B, C, H_new, W_new)
            self.fake_H = self.netG(var_L, mask)
            self.fake_H = F.interpolate(self.fake_H, size=[H, W], mode='bilinear')

            del var_L
            del mask
            torch.cuda.empty_cache()

        self.netG.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()



        dark = self.var_L
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        if not (len(self.nf.shape) == 4):
            self.nf = self.nf.unsqueeze(dim=0)
        light = self.nf
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)
        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)
        mask[mask <= 0.5] = 0.0
        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()
        mask = mask.repeat(1, 3, 1, 1)
        out_dict['rlt3'] = mask[0].float().cpu()


        # self.gray=rgb2gray(self.var_L)
        # out_dict['rlt3'] = self.gray.detach()[0].float().cpu()
        # out_dict['rlt4'] = lf.detach()[0].float().cpu()


        out_dict['GT'] = self.real_H.detach()[0].float().cpu()                       #测试使用
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()

        # if need_GT:
        #     out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        del mask
        del self.var_L
        del self.fake_H
        torch.cuda.empty_cache()
        return out_dict


    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
