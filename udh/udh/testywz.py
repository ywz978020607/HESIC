# python testywz.py /home/ywz/database/aftercut512/test/
import torch
import argparse
import kornia
import imageio
import numpy as np
import torch.nn as nn

# from train import HomographyModel
import os,glob
import os.path as osp


#修改
from dataset import SyntheticDataset, MEAN, STD
from dataset import SyntheticDataset, safe_collate

from model import Net, photometric_loss

pic_size = 256
patch_size = 128  #最好别变，可以改pic，可以获取角点进行缩放后求H

class HomographyModel(nn.Module):
    def __init__(self):
        super(HomographyModel, self).__init__()
        self.model = Net(patch_size=patch_size)

    def forward(self, a, b):
        return self.model(a, b)


def tensors_to_gif(a, b, name):
    a = a.permute(1, 2, 0).numpy()
    b = b.permute(1, 2, 0).numpy()
    imageio.mimsave(name, [a, b], duration=1)


@torch.no_grad()
def main(args):
    if args.resume !="":
        model = HomographyModel.load_from_checkpoint(args.resume)
    else:
        # model_dir = 'lightning_logs/version*'
        # model_dir_list = sorted(glob.glob(model_dir))
        # model_dir = model_dir_list[-1]
        # model_path = osp.join(model_dir, "checkpoints", "*.ckpt")
        # model_path_list = sorted(glob.glob(model_path))
        model_path_list = sorted(glob.glob("*.ckpt"))
        print(model_path_list)
        if len(model_path_list) > 0:
            model_path = model_path_list[-1]
            print(model_path)
            # model = HomographyModel.load_from_checkpoint(model_path)
            model = HomographyModel() #test专用，内部重新定义精简版的类
            model_old = torch.load(model_path, map_location=lambda storage, loc: storage)
            # print(model_old.keys())
            # net.load_state_dict(torch.load('path/params.pkl'))
            model.load_state_dict(model_old['state_dict'])
            print(model_path)
            print("model loaded.")
        else:
            # raise ValueError(f'No load model!')  #raise Error
            print("No load model!")
            model = HomographyModel()  # test专用，内部重新定义精简版的类

    model.eval()  #不训练

    test_set = SyntheticDataset(args.test_path, rho=args.rho, filetype=args.filetype,pic_size=pic_size,patch_size=patch_size)

    #clear last output
    last_output = "figures/*"
    os.system("rm "+last_output)
    print('clear last ok.')
    for i in range(args.n):
        img_a,img_b, patch_a, patch_b, corners, delta = test_set[i]
        # after unsqueeze to save
        # tensors_to_gif(patch_a, patch_b, f"figures/input_{i}.gif")
        tensors_to_gif(img_a, img_b, f"figures/input_{i}.gif")

        #add
        img_a = img_a.unsqueeze(0)
        img_b = img_b.unsqueeze(0)


        ##
        patch_a = patch_a.unsqueeze(0)
        patch_b = patch_b.unsqueeze(0)
        corners = corners.unsqueeze(0)
        # print(corners)
        corners = corners - corners[:, 0].view(-1, 1, 2)
        # print(corners)
        delta_hat = model(patch_a, patch_b)
        corners_hat = corners + delta_hat
        #获取h
        h = kornia.get_perspective_transform(corners, corners_hat)
        h_inv = torch.inverse(h)

        # print(patch_a)
        # print(delta_hat)
        # print(corners.dtype)
        # print(corners_hat)
        print(h)
        print(h_inv)
        raise ValueError("stop")


        patch_b_hat = kornia.warp_perspective(img_a, h_inv, (patch_a.shape[-2],patch_a.shape[-1]))  #128 最初设置 #注意，用img_a / patch_a
        img_b_hat = kornia.warp_perspective(img_a, h_inv, (img_a.shape[-2],img_a.shape[-1]))
        #输出

        tensors_to_gif(patch_b_hat[0], patch_b[0], f"figures/output_patch{i}.gif")
        tensors_to_gif(img_b_hat[0], img_b[0], f"figures/output_{i}.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",type=str, default="")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--rho", type=int, default=20, help="amount to perturb corners")
    parser.add_argument("--n", type=int, default=5, help="number of images to test")
    parser.add_argument("--filetype", default=".png")
    parser.add_argument("test_path", help="path to test images")
    args = parser.parse_args()
    main(args)
