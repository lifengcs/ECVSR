from torch.autograd import Variable
from torch.utils.data import DataLoader
from modules import SOFVSR
from data_utils import TrainsetLoader, OFR_loss
import torch.backends.cudnn as cudnn
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from Net import Net
from loss import CharbonnierLoss

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degradation", type=str, default='RA')
    parser.add_argument("--QP", type=int, default=22)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--patch_size', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_iters', type=int, default=40100, help='number of iterations to train')
    parser.add_argument('--trainset_dir', type=str, default='training/dataset/path')
    return parser.parse_args()

def main(cfg):
    # model
    net = Net(cfg, in_nc=1, out_nc=1, nf=128, nframes=7)
    if cfg.gpu_mode:
        net = nn.DataParallel(net)
        net.cuda()
    cudnn.benchmark = True
    train_set = TrainsetLoader(cfg)
    train_loader = DataLoader(train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)

    # train
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4)
    milestones = [10000, 20000, 30000, 40000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    loss_list = []
    for idx_iter, (LR, HR) in enumerate(train_loader):
        scheduler.step()

        # data
        b, n_frames, h_lr, w_lr = LR.size()
        idx_center = (n_frames - 1) // 2

        LR, HR = Variable(LR), Variable(HR)
        if cfg.gpu_mode:
            LR = LR.cuda()
            HR = HR.cuda()
        LR = LR.view(b, -1, 1, h_lr, w_lr)
        HR = HR.view(b, -1, 1, h_lr * cfg.scale, w_lr * cfg.scale)
        sr_out, sr_rf = net(LR)

        # loss
        loss = CharbonnierLoss(sr_out, HR[:, idx_center, :, :, :]) + CharbonnierLoss(sr_rf, HR[:, idx_center, :, :, :])
        loss_list.append(loss.data.cpu())

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Iteration---%6d,   loss---%f' % (idx_iter + 1, np.array(loss_list).mean()))

        # save checkpoint
        if idx_iter % 500 == 0:
            # print('Iteration---%6d,   loss---%f' % (idx_iter + 1, np.array(loss_list).mean()))
            save_path = 'models/' + cfg.degradation + '_x' + str(cfg.scale) + '/QP' + str(cfg.QP)
            save_name = cfg.degradation + '_QP' + str(cfg.QP) + '_iter' + str(idx_iter) + '.pth'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(net.state_dict(), save_path + '/' + save_name)
            loss_list = []


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)






