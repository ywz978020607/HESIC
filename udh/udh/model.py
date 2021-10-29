import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import os
from torchvision import transforms

#save for test
def save_pic(data,path):
    if data.device.type!='cpu':
        reimage = data.cpu().clone()
    else:
        reimage = data
    reimage = reimage.squeeze(0)
    reimage = transforms.ToPILImage()(reimage) #PIL格式
    reimage.save(path)

def photometric_loss(delta, img_a, patch_b, corners):
    corners_hat = corners + delta

    # in order to apply transform and center crop,
    # subtract points by top-left corner (corners[N, 0])
    # print("ori_corners:",corners[0:1])
    corners = corners - corners[:, 0].view(-1, 1, 2)  #关键！！！！区分大变换还是小变换！！！！！！！！！！！！！

    h = kornia.get_perspective_transform(corners, corners_hat)

    h_inv = torch.inverse(h)  #求逆矩阵
    # patch_b_hat = kornia.warp_perspective(img_a, h_inv, (128, 128))
    patch_b_hat = kornia.warp_perspective(img_a, h_inv, (patch_b.shape[-2],patch_b.shape[-1]))
    # patch_b_hat2 = kornia.warp_perspective(img_a, h_inv, (img_a.shape[-2],img_a.shape[-1]))
    # print("corners:",corners[0:1])
    # print("corners_hat:",corners_hat[0:1])
    # print("H:",h[0:1])
    # print(img_a.size())
    # print(patch_b.size())
    # print(patch_b_hat.size())
    #
    # save_pic(img_a[0:1, :], "img_a.png")
    # save_pic(patch_b[0:1, :], "patch_b.png")
    # save_pic(patch_b_hat[0:1, :], "patch_b_hat.png")
    # save_pic(patch_b_hat2[0:1, :], "patch_b_hat2.png")
    # raise  ValueError("print size")

    return F.l1_loss(patch_b_hat, patch_b)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Block(nn.Module):
    def __init__(self, inchannels, outchannels, batch_norm=False, pool=True):
        super(Block, self).__init__()
        layers = []
        layers.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        layers.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Net(nn.Module):
    def __init__(self, batch_norm=False,patch_size=128):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
            Block(2, 64, batch_norm),
            Block(64, 64, batch_norm),
            Block(64, 128, batch_norm),
            Block(128, 128, batch_norm, pool=False),
        )
        self.fc = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),
            # nn.Linear(128 * 16 * 16, 1024),
            nn.Linear(128 * (patch_size//8) * (patch_size//8), 1024), #呼应128/8 = 16 -> 512/8 = 64
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 4 * 2),
        )


    def forward(self, a, b):
        x = torch.cat((a, b), dim=1)  # combine two images in channel dimension
        x = self.cnn(x)
        x = self.fc(x)
        delta = x.view(-1, 4, 2)
        return delta   #返回两个patch块的delta 4*2

    def get_h(self, a, b,corners):
        x = torch.cat((a, b), dim=1)  # combine two images in channel dimension
        x = self.cnn(x)
        x = self.fc(x)
        delta = x.view(-1, 4, 2)

        corners_hat = corners + delta
        # 获取h
        h = kornia.get_perspective_transform(corners, corners_hat)
        h_inv = torch.inverse(h)

        return h_inv
