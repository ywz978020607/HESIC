# test3.py  test2:PSNR计算后再平均-=> for 独立交叉质量增强 搭配newtrain6.py
# 单gpu版  和训练代码一样，只是把训练部分注释
#python test3.py -d "/home/ywz/database/aftercut512"  --seed 0 --cuda 0 --patch-size 512 512 --batch-size 1 --test-batch-size 1
#python test3.py -d "/home/ywz/database/KITTI2"  --seed 0 --cuda 0 --patch-size 320 1216 --batch-size 1 --test-batch-size 1
#cpu版
#python test3.py -d "/home/ywz/database/aftercut512"  --seed 0  --patch-size 512 512 --batch-size 1 --test-batch-size 1
import argparse
import math
import random
import shutil
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.autograd import Variable

from torch.utils.data import DataLoader

from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.layers import GDN
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#net defination

from newnet9 import *

def mse2psnr(mse):
    # 根据Hyper论文中的内容，将MSE->psnr(db)
    # return 10*math.log10(255*255/mse)
    return 10 * math.log10(1/ mse) #???
#psnr calculate
def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

################################################################

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target1,target2):
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

        # 计算误差
        # out['bpp_loss'] = sum(
        #     (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        #     for likelihoods in output['likelihoods'].values())
        out['mse_loss'] = self.mse(output['x1_hat'], target1) + self.mse(output['x2_hat'], target2)        #end to end
        # out['bpp1'] = (torch.log(output['likelihoods']['y1']).sum() / (-math.log(2) * num_pixels)) + (
        #         torch.log(output['likelihoods']['z1']).sum() / (-math.log(2) * num_pixels))
        # out['bpp2'] = (torch.log(output['likelihoods']['y2']).sum() / (-math.log(2) * num_pixels)) + (
        #         torch.log(output['likelihoods']['z2']).sum() / (-math.log(2) * num_pixels))

        out['loss'] = self.lmbda * 255**2 * out['mse_loss'] #+ out['bpp_loss']

        out['ms_ssim1'] = ms_ssim(output['x1_hat'], target1, data_range=1, size_average=False)[0]  # (N,)
        out['ms_ssim2'] = ms_ssim(output['x2_hat'], target2, data_range=1, size_average=False)[0]
        out['ms_ssim'] = (out['ms_ssim1']+out['ms_ssim2'])/2
        out['psnr1'] = mse2psnr(self.mse(output['x1_hat'], target1))
        out['psnr2'] = mse2psnr(self.mse(output['x2_hat'], target2))

        return out


class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def test_epoch(epoch, test_dataloader, model,model2, criterion):
    model.eval()
    model2.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    ssim_loss = AverageMeter()
    ssim_loss1 = AverageMeter()
    ssim_loss2 = AverageMeter()

    psnr1 = AverageMeter()
    psnr2 = AverageMeter()
    bpp1 = AverageMeter()
    bpp2 = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d1 = d[0].to(device)
            d2 = d[1].to(device)
            h_matrix = d[2].to(device)

            out_net = model(d1,d2,h_matrix)

            out_net2 = model2(out_net['x1_hat'], out_net['x2_hat'], h_matrix)
            # out_net['x1_hat'] = out_net2['x1_hat']
            # out_net['x2_hat'] = out_net2['x2_hat']

            out_criterion = criterion(out_net2, d1, d2)
            # out_criterion = criterion(out_net, d1,d2)

            aux_loss.update(model.aux_loss())
            # bpp_loss.update(out_criterion['bpp_loss'])
            loss.update(out_criterion['loss'])
            mse_loss.update(out_criterion['mse_loss'])
            ssim_loss.update(out_criterion['ms_ssim'])  # 已除2
            ssim_loss1.update(out_criterion['ms_ssim1'])  # 已除2
            ssim_loss2.update(out_criterion['ms_ssim2'])  # 已除2

            psnr1.update(out_criterion['psnr1'])
            psnr2.update(out_criterion['psnr2'])
            # bpp1.update(out_criterion['bpp1'])
            # bpp2.update(out_criterion['bpp2'])

    print(f'Test epoch {epoch}: Average losses:'
          f'\tTime: {time.strftime("%Y-%m-%d %H:%M:%S")} |'
          f'\tLoss: {loss.val:.3f} |'
          f'\tMSE loss: {mse_loss.val:.4f} |'
          f'\tPSNR (dB): {(psnr1.val+psnr2.val)/2:.3f} |'  #平均一张图的PSNR
          f'\tMS-SSIM: {ssim_loss.val:.4f} |'  #已除2，相加时候便除了2
          f'\tMS-SSIM1: {ssim_loss1.val:.4f} |'
          f'\tMS-SSIM2: {ssim_loss2.val:.4f} |'
          f'\tPSNR1: {psnr1.val:.3f} |'
          f'\tPSNR2: {psnr2.val:.3f} \n'
          )

    return loss.val


def save_checkpoint(state, is_best, filename='second_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'second_checkpoint_best_loss.pth.tar')


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Example training script')
    # yapf: disable
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        help='Training dataset')
    parser.add_argument(
        '-e',
        '--epochs',
        default=100,
        type=int,
        help='Number of epochs (default: %(default)s)')
    parser.add_argument(
        '-lr',
        '--learning-rate',
        default=1e-4,
        type=float,
        help='Learning rate (default: %(default)s)')
    parser.add_argument(
        '-n',
        '--num-workers',
        type=int,
        default= 3,
        help='Dataloaders threads (default: %(default)s)')
    parser.add_argument(
        '--lambda',
        dest='lmbda',
        type=float,
        default=1e-2,
        help='Bit-rate distortion parameter (default: %(default)s)')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: %(default)s)')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=64,
        help='Test batch size (default: %(default)s)')
    parser.add_argument(
        '--aux-learning-rate',
        default=1e-3,
        help='Auxiliary loss learning rate (default: %(default)s)')
    parser.add_argument(
        '--patch-size',
        type=int,
        nargs=2,
        default=(256, 256),
        help='Size of the patches to be cropped (default: %(default)s)')
    parser.add_argument(
        '--cuda',
        type=int,
        default=-1,
        help='Use cuda')
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save model to disk')
    parser.add_argument(
        '--logfile',
        type=str,
        default="train_log.txt",
        help='logfile_name')
    parser.add_argument(
        '--seed',
        type=float,
        help='Set random seed for reproducibility')
    # yapf: enable
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # train_transforms = transforms.Compose(
    #     [transforms.RandomCrop(args.patch_size),
    #      transforms.ToTensor()])
    #
    # test_transforms = transforms.Compose(
    #     [transforms.CenterCrop(args.patch_size),
    #      transforms.ToTensor()])
    train_transforms = transforms.Compose(
        [transforms.ToTensor()])

    test_transforms = transforms.Compose(
        [transforms.ToTensor()])

    train_dataset = ImageFolder(args.dataset,
                                split='train',
                                patch_size=args.patch_size,
                                transform=train_transforms)
    test_dataset = ImageFolder(args.dataset,
                               split='test',
                               patch_size=args.patch_size,
                               transform=test_transforms)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  pin_memory=False)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.test_batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 pin_memory=False)


    device = 'cuda' if (torch.cuda.is_available() and args.cuda!=-1) else 'cpu'
    print(device)
    if device=='cuda':
        torch.cuda.set_device(args.cuda)
    print('temp gpu device number:')
    print(torch.cuda.current_device())
    #net assign
    # with torch.autograd.set_detect_anomaly(True): #for debug gradient
    # net = DSIC(N=128,M=192,F=21,C=32,K=5) #(N=128,M=192,F=21,C=32,K=5)
    net = HSIC(N=128, M=192, K=5)
    net2 = Independent_EN()  #交叉质量增强
    #也可用GMM_together() 调用一个网络包括整体  分开调用方便测试溶解效果


    # net = HSIC(N=128, M=192, K=15)
    # 加载最新模型继续训练
    if os.path.exists("checkpoint_best_loss.pth.tar"):
        model = torch.load('checkpoint_best_loss.pth.tar', map_location=lambda storage, loc: storage)
        model.keys()
        # net.load_state_dict(torch.load('path/params.pkl'))
        net.load_state_dict(model['state_dict'])
        print("load model ok")
    else:
        print("train from none")

    # 加载最新模型继续训练
    if os.path.exists("second_checkpoint_best_loss.pth.tar"):
        model = torch.load('second_checkpoint_best_loss.pth.tar', map_location=lambda storage, loc: storage)
        model.keys()
        # net.load_state_dict(torch.load('path/params.pkl'))
        net2.load_state_dict(model['state_dict'])
        print("2load model ok")
    else:
        print("2train from none")

    net = net.to(device)
    net2 = net2.to(device)
    optimizer = optim.Adam(net2.parameters(), lr=args.learning_rate)
    aux_optimizer = optim.Adam(net.aux_parameters(), lr=args.aux_learning_rate)
    print("lambda:", args.lmbda)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    for epoch in [0]:  # 只跑一次
        loss = test_epoch(epoch, test_dataloader, net, net2, criterion)

if __name__ == '__main__':
    main(sys.argv[1:])