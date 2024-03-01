from PIL import Image
from torch.utils.data.dataset import Dataset
from modules import optical_flow_warp
import numpy as np
import os
import torch
import random
import cv2

class TrainsetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainsetLoader).__init__()
        self.trainset_dir = cfg.trainset_dir
        self.scale = cfg.scale
        self.patch_size = cfg.patch_size
        self.n_iters = cfg.n_iters * cfg.batch_size
        self.video_list = os.listdir(cfg.trainset_dir)
        self.degradation = cfg.degradation
        self.qp = cfg.QP

    def __getitem__(self, idx):
        idx_video = random.randint(0, len(self.video_list)-1)
        idx_frame = random.randint(0, 25)                           # #frames of training videos is 31, 31-3=28
        lr_dir = self.trainset_dir + self.video_list[idx_video] + '/' + str(self.degradation) + '/QP' + str(self.qp) + '/'
        hr_dir = self.trainset_dir + self.video_list[idx_video] + '/GT/'

        # read HR & LR frames
        LR0 = Image.open(lr_dir + str(idx_frame).zfill(5) + '.png')
        LR1 = Image.open(lr_dir + str(idx_frame + 1).zfill(5) + '.png')
        LR2 = Image.open(lr_dir + str(idx_frame + 2).zfill(5) + '.png')
        LR3 = Image.open(lr_dir + str(idx_frame + 3).zfill(5) + '.png')
        LR4 = Image.open(lr_dir + str(idx_frame + 4).zfill(5) + '.png')
        LR5 = Image.open(lr_dir + str(idx_frame + 5).zfill(5) + '.png')
        LR6 = Image.open(lr_dir + str(idx_frame + 6).zfill(5) + '.png')

        HR0 = Image.open(hr_dir + str(idx_frame).zfill(5) + '.png')
        HR1 = Image.open(hr_dir + str(idx_frame + 1).zfill(5) + '.png')
        HR2 = Image.open(hr_dir + str(idx_frame + 2).zfill(5) + '.png')
        HR3 = Image.open(hr_dir + str(idx_frame + 3).zfill(5) + '.png')
        HR4 = Image.open(hr_dir + str(idx_frame + 4).zfill(5) + '.png')
        HR5 = Image.open(hr_dir + str(idx_frame + 5).zfill(5) + '.png')
        HR6 = Image.open(hr_dir + str(idx_frame + 6).zfill(5) + '.png')

        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        LR1 = np.array(LR1, dtype=np.float32) / 255.0
        LR2 = np.array(LR2, dtype=np.float32) / 255.0
        LR3 = np.array(LR3, dtype=np.float32) / 255.0
        LR4 = np.array(LR4, dtype=np.float32) / 255.0
        LR5 = np.array(LR5, dtype=np.float32) / 255.0
        LR6 = np.array(LR6, dtype=np.float32) / 255.0


        HR0 = np.array(HR0, dtype=np.float32) / 255.0
        HR1 = np.array(HR1, dtype=np.float32) / 255.0
        HR2 = np.array(HR2, dtype=np.float32) / 255.0
        HR3 = np.array(HR3, dtype=np.float32) / 255.0
        HR4 = np.array(HR4, dtype=np.float32) / 255.0
        HR5 = np.array(HR5, dtype=np.float32) / 255.0
        HR6 = np.array(HR6, dtype=np.float32) / 255.0

        # crop patchs randomly
        HR0, HR1, HR2, HR3, HR4, HR5, HR6, LR0, LR1, LR2, LR3, LR4, LR5, LR6 = \
            random_crop(HR0, HR1, HR2, HR3, HR4, HR5, HR6,
                        LR0, LR1, LR2, LR3, LR4, LR5, LR6,
                        self.patch_size, self.scale)

        HR0 = HR0[:, :, np.newaxis]
        HR1 = HR1[:, :, np.newaxis]
        HR2 = HR2[:, :, np.newaxis]
        HR3 = HR3[:, :, np.newaxis]
        HR4 = HR4[:, :, np.newaxis]
        HR5 = HR5[:, :, np.newaxis]
        HR6 = HR6[:, :, np.newaxis]

        LR0 = LR0[:, :, np.newaxis]
        LR1 = LR1[:, :, np.newaxis]
        LR2 = LR2[:, :, np.newaxis]
        LR3 = LR3[:, :, np.newaxis]
        LR4 = LR4[:, :, np.newaxis]
        LR5 = LR5[:, :, np.newaxis]
        LR6 = LR6[:, :, np.newaxis]

        HR = np.concatenate((HR0, HR1, HR2, HR3, HR4, HR5, HR6), axis=2)
        LR = np.concatenate((LR0, LR1, LR2, LR3, LR4, LR5, LR6), axis=2)

        # data augmentation
        LR, HR = augmentation()(LR, HR)

        return toTensor(LR), toTensor(HR)

    def __len__(self):
        return self.n_iters


class TestsetLoader(Dataset):
    def __init__(self, cfg, video_name):
        super(TestsetLoader).__init__()
        self.dataset_dir = cfg.testset_dir + '/' + video_name
        self.degradation = cfg.degradation
        self.scale = cfg.scale
        self.frame_list = os.listdir(self.dataset_dir)

    def __getitem__(self, idx):
        dir = self.dataset_dir + '/'
        LR0 = Image.open(dir + '/' + str(idx + 1).rjust(3, '0') + '.png')
        LR1 = Image.open(dir + '/' + str(idx + 2).rjust(3, '0') + '.png')
        LR2 = Image.open(dir + '/' + str(idx + 3).rjust(3, '0') + '.png')
        LR3 = Image.open(dir + '/' + str(idx + 4).rjust(3, '0') + '.png')
        LR4 = Image.open(dir + '/' + str(idx + 5).rjust(3, '0') + '.png')
        LR5 = Image.open(dir + '/' + str(idx + 6).rjust(3, '0') + '.png')
        LR6 = Image.open(dir + '/' + str(idx + 7).rjust(3, '0') + '.png')
        W, H = LR1.size

        LR3_bicubic = LR3.resize((W * self.scale, H * self.scale), Image.BICUBIC)
        LR3_bicubic = np.array(LR3_bicubic, dtype=np.float32) / 255.0

        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        LR1 = np.array(LR1, dtype=np.float32) / 255.0
        LR2 = np.array(LR2, dtype=np.float32) / 255.0
        LR3 = np.array(LR3, dtype=np.float32) / 255.0
        LR4 = np.array(LR4, dtype=np.float32) / 255.0
        LR5 = np.array(LR5, dtype=np.float32) / 255.0
        LR6 = np.array(LR6, dtype=np.float32) / 255.0

        # extract Y channel for LR inputs
        LR0_y, _, _ = rgb2ycbcr(LR0)
        LR1_y, _, _ = rgb2ycbcr(LR1)
        LR2_y, _, _ = rgb2ycbcr(LR2)
        LR3_y, _, _ = rgb2ycbcr(LR3)
        LR4_y, _, _ = rgb2ycbcr(LR4)
        LR5_y, _, _ = rgb2ycbcr(LR5)
        LR6_y, _, _ = rgb2ycbcr(LR6)

        LR0_y = LR0_y[:, :, np.newaxis]
        LR1_y = LR1_y[:, :, np.newaxis]
        LR2_y = LR2_y[:, :, np.newaxis]
        LR3_y = LR3_y[:, :, np.newaxis]
        LR4_y = LR4_y[:, :, np.newaxis]
        LR5_y = LR5_y[:, :, np.newaxis]
        LR6_y = LR6_y[:, :, np.newaxis]
        LR = np.concatenate((LR0_y, LR1_y, LR2_y, LR3_y , LR4_y, LR5_y, LR6_y), axis=2)

        LR = toTensor(LR)

        # generate Cr, Cb channels using bicubic interpolation
        _, SR_cb, SR_cr = rgb2ycbcr(LR3_bicubic)

        return LR, SR_cb, SR_cr

    def __len__(self):
        return len(self.frame_list) - 6


class augmentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[:, ::-1, :]
            target = target[:, ::-1, :]
        if random.random()<0.5:
            input = input[::-1, :, :]
            target = target[::-1, :, :]
        if random.random()<0.5:
            # print(input.shape)
            input = input.transpose(1, 0, 2)
            target = target.transpose(1, 0, 2)
        return np.ascontiguousarray(input), np.ascontiguousarray(target)


def random_crop(HR0, HR1, HR2, HR3, HR4, HR5, HR6,
                LR0, LR1, LR2, LR3, LR4, LR5, LR6,
                patch_size_lr, scale):
    h_hr, w_hr = HR0.shape
    h_lr = h_hr // scale
    w_lr = w_hr // scale
    idx_h = random.randint(10, h_lr - patch_size_lr - 10)
    idx_w = random.randint(10, w_lr - patch_size_lr - 10)

    h_start_hr = (idx_h - 1) * scale
    h_end_hr = (idx_h - 1 + patch_size_lr) * scale
    w_start_hr = (idx_w - 1) * scale
    w_end_hr = (idx_w - 1 + patch_size_lr) * scale

    h_start_lr = idx_h - 1
    h_end_lr = idx_h - 1 + patch_size_lr
    w_start_lr = idx_w - 1
    w_end_lr = idx_w - 1 + patch_size_lr

    HR0 = HR0[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    HR1 = HR1[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    HR2 = HR2[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    HR3 = HR3[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    HR4 = HR4[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    HR5 = HR5[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    HR6 = HR6[h_start_hr:h_end_hr, w_start_hr:w_end_hr]

    LR0 = LR0[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    LR1 = LR1[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    LR2 = LR2[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    LR3 = LR3[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    LR4 = LR4[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    LR5 = LR5[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    LR6 = LR6[h_start_lr:h_end_lr, w_start_lr:w_end_lr]

    return HR0, HR1, HR2, HR3, HR4, HR5, HR6, \
           LR0, LR1, LR2, LR3, LR4, LR5, LR6


def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img


def rgb2ycbcr(img_rgb):
    ## the range of img_rgb should be (0, 1)
    img_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] + 16 / 255.0
    img_cb = -0.148 * img_rgb[:, :, 0] - 0.291 * img_rgb[:, :, 1] + 0.439 * img_rgb[:, :, 2] + 128 / 255.0
    img_cr = 0.439 * img_rgb[:, :, 0] - 0.368 * img_rgb[:, :, 1] - 0.071 * img_rgb[:, :, 2] + 128 / 255.0
    return img_y, img_cb, img_cr


def ycbcr2rgb(img_ycbcr):
    ## the range of img_ycbcr should be (0, 1)
    img_r = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 1.596 * (img_ycbcr[:, :, 2] - 128 / 255.0)
    img_g = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) - 0.392 * (img_ycbcr[:, :, 1] - 128 / 255.0) - 0.813 * (img_ycbcr[:, :, 2] - 128 / 255.0)
    img_b = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 2.017 * (img_ycbcr[:, :, 1] - 128 / 255.0)
    img_r = img_r[:, :, np.newaxis]
    img_g = img_g[:, :, np.newaxis]
    img_b = img_b[:, :, np.newaxis]
    img_rgb = np.concatenate((img_r, img_g, img_b), 2)
    return img_rgb


def rgb2y(img_rgb):
    ## the range of img_rgb should be (0, 1)
    image_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] +16 / 255.0
    return image_y


def OFR_loss(x0, x1, optical_flow):
    warped = optical_flow_warp(x0, optical_flow)
    loss = torch.mean(torch.abs(x1 - warped)) + 0.1 * L1_regularization(optical_flow)
    return loss


def L1_regularization(image):
    b, _, h, w = image.size()
    reg_x_1 = image[:, :, 0:h-1, 0:w-1] - image[:, :, 1:, 0:w-1]
    reg_y_1 = image[:, :, 0:h-1, 0:w-1] - image[:, :, 0:h-1, 1:]
    reg_L1 = torch.abs(reg_x_1) + torch.abs(reg_y_1)
    return torch.sum(reg_L1) / (b*(h-1)*(w-1))
