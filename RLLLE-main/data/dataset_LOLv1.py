import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import torch.nn.functional as F
import random
import cv2
import numpy as np
import glob
import os
import functools
from torchvision.transforms import  Resize
import cv2
class VideoSameSizeDataset(data.Dataset):
    def __init__(self, opt):
        super(VideoSameSizeDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root ,self.DAY_root ,self.MASK_root= opt['dataroot_GT'], opt['dataroot_LQ'], opt['dataroot_DAY'], opt['dataroot_MASK']
        self.data_type = self.opt['data_type']
        self.data_info = {'path_LQ': [], 'path_GT': [],'path_DAY': [], 'path_MASK': [], 'folder': [],'folder_mask': [],'folder_day': [],  'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        # Generate data info and cache data
        self.imgs_LQ, self.imgs_GT, self.imgs_DAY,self.imgs_MASK = {},{},{},{}

        subfolders_LQ = util.glob_file_list(self.LQ_root)
        subfolders_GT = util.glob_file_list(self.GT_root)
        subfolders_DAY = util.glob_file_list(self.DAY_root)

        subfolders_MASK = util.glob_file_list(self.MASK_root)



        count = 0
        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            subfolder_name = osp.basename(subfolder_LQ)

            img_paths_LQ = [subfolder_LQ]
            img_paths_GT = [subfolder_GT]

            max_idx = len(img_paths_LQ)
            self.data_info['path_LQ'].extend(img_paths_LQ)  # list of path str of images
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            self.data_info['idx'].append('{}/{}'.format(count, len(subfolder_LQ)))

            if self.cache_data:
                self.imgs_LQ[subfolder_name] = img_paths_LQ
                self.imgs_GT[subfolder_name] = img_paths_GT
        for subfolder_MASK in subfolders_MASK:
            subfolder_name = osp.basename(subfolder_MASK)

            img_paths_MASK = [subfolder_MASK]


            max_idx = len(img_paths_MASK)
            self.data_info['path_MASK'].extend(img_paths_MASK)  # list of path str of images
            self.data_info['folder_mask'].extend([subfolder_name] * max_idx)
            self.data_info['idx'].append('{}/{}'.format(count, len(subfolder_MASK)))

            if self.cache_data:
                self.imgs_MASK[subfolder_name] = img_paths_MASK
        for subfolder_DAY in subfolders_DAY:
            subfolder_name = osp.basename(subfolder_DAY)

            img_paths_DAY = [subfolder_DAY]

            max_idx = len(img_paths_DAY)
            self.data_info['path_DAY'].extend(img_paths_DAY)  # list of path str of images
            self.data_info['folder_day'].extend([subfolder_name] * max_idx)
            self.data_info['idx'].append('{}/{}'.format(count, len(subfolder_DAY)))

            if self.cache_data:
                self.imgs_DAY[subfolder_name] = img_paths_DAY

    def __getitem__(self, index):


        folder = self.data_info['folder'][index]
        folder_mask = self.data_info['folder_mask'][index]
        folder_day = self.data_info['folder_day'][index]
        img_LQ_path = self.imgs_LQ[folder][0]
        img_GT_path = self.imgs_GT[folder][0]


        img_MASK_path = self.imgs_MASK[folder_mask][0]

        img_DAY_path = self.imgs_DAY[folder_day][0]


        img_LQ_path = [img_LQ_path]
        img_GT_path = [img_GT_path]
        img_MASK_path =[img_MASK_path]
        img_DAY_path = [img_DAY_path]



        if self.opt['phase'] == 'train':


            img_LQ = util.read_img_seq(img_LQ_path)
            img_LQ_HIST= util.read_img_seq_hist(img_LQ_path)
           
            img_GT = util.read_img_seq(img_GT_path)

            img_DAY = cv2.imread(img_DAY_path[0])
            img_DAY = img_DAY.astype(np.float32) / 255.
            img_DAY = np.expand_dims(img_DAY, axis=0)
            img_DAY = torch.from_numpy(np.ascontiguousarray(np.transpose(img_DAY, (0, 3, 1, 2)))).float()
            img_DAY = F.interpolate(img_DAY, size=[400, 600], mode='nearest')
            img_MASK=np.load(img_MASK_path[0])


            img_MASK=np.expand_dims(img_MASK, axis=0)


            img_MASK = torch.from_numpy(np.ascontiguousarray(np.transpose(img_MASK, (0, 3, 1, 2)))).float()
            img_MASK = F.interpolate(img_MASK, size=[400, 600], mode='nearest')
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]
            img_DAY = img_DAY[0]
            img_LQ_HIST=img_LQ_HIST[0]
            img_MASK = img_MASK[0]




            LQ_size = self.opt['LQ_size']
            GT_size = self.opt['GT_size']
            MASK_size = self.opt['LQ_size']

            _, H, W = img_GT.shape  # real img size

            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_LQ = img_LQ[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
            img_LQ_HIST = img_LQ_HIST[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
            img_GT = img_GT[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
            img_DAY = img_DAY[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]

            img_MASK = img_MASK[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]



            img_LQ_l = [img_LQ]
            img_LQ_l.append(img_GT)
            img_LQ_l.append(img_MASK)
            img_LQ_l.append(img_LQ_HIST)
            img_LQ_l.append(img_DAY)

            rlt = util.augment_torch(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ = rlt[0]
            img_GT = rlt[1]
            img_MASK = rlt[2]
            img_LQ_HIST=rlt[3]
            img_DAY = rlt[4]


        elif self.opt['phase'] == 'test':

            img_LQ = util.read_img_seq(img_LQ_path, self.opt['train_size'])
            # img_LQ = util.read_img_seq(img_LQ_path)
            img_DAY = cv2.imread(img_DAY_path[0])
            img_DAY = img_DAY.astype(np.float32) / 255.
            img_DAY = np.expand_dims(img_DAY, axis=0)
            img_DAY = torch.from_numpy(np.ascontiguousarray(np.transpose(img_DAY, (0, 3, 1, 2)))).float()
            img_DAY = F.interpolate(img_DAY, size=[400, 600], mode='nearest')
            img_GT = util.read_img_seq(img_GT_path)
            img_GT = util.read_img_seq(img_GT_path, self.opt['train_size'])
            img_LQ_HIST = util.read_img_seq_hist(img_LQ_path, self.opt['train_size'])
            img_LQ_HIST = util.read_img_seq_hist(img_LQ_path)

            img_MASK = np.load(img_MASK_path[0])
            img_MASK = np.expand_dims(img_MASK, axis=0)
            img_MASK = torch.from_numpy(np.ascontiguousarray(np.transpose(img_MASK, (0, 3, 1, 2)))).float()
            img_MASK = F.interpolate(img_MASK, size=[400, 600], mode='nearest')

            img_LQ = img_LQ[0]
            img_GT = img_GT[0]
            img_DAY = img_DAY[0]
            img_MASK = img_MASK[0]
            img_LQ_HIST = img_LQ_HIST[0]




        else:
            img_LQ = util.read_img_seq(img_LQ_path, self.opt['train_size'])
            img_GT = util.read_img_seq(img_GT_path, self.opt['train_size'])

            img_DAY = cv2.imread(img_DAY_path[0])
            img_DAY = img_DAY.astype(np.float32) / 255.
            img_DAY = np.expand_dims(img_DAY, axis=0)
            img_DAY = torch.from_numpy(np.ascontiguousarray(np.transpose(img_DAY, (0, 3, 1, 2)))).float()
            img_DAY = F.interpolate(img_DAY, size=[400, 600], mode='nearest')




            img_LQ_HIST = util.read_img_seq_hist(img_LQ_path, self.opt['train_size'])



            img_MASK = np.load(img_MASK_path[0])
            img_MASK = np.expand_dims(img_MASK, axis=0)
            img_MASK = torch.from_numpy(np.ascontiguousarray(np.transpose(img_MASK, (0, 3, 1, 2)))).float()

            img_MASK = F.interpolate(img_MASK, size=[400, 600], mode='nearest')

            img_LQ = img_LQ[0]
            img_GT = img_GT[0]
            img_DAY = img_DAY[0]
            img_LQ_HIST = img_LQ_HIST[0]


            img_MASK = img_MASK




        img_nf = img_LQ.permute(1, 2, 0).numpy() * 255.0
        img_nf = cv2.blur(img_nf, (5, 5))
        img_nf = img_nf * 1.0 / 255.0
        img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1)


        return {
            'LQs': img_LQ,
            'GT': img_GT,
            'DAY': img_DAY,
            'MASK': img_MASK,
            'nf': img_nf,
            'img_LQ_HIST':img_LQ_HIST,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': 0
        }

    def __len__(self):
        return len(self.data_info['path_LQ'])
