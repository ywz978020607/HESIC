import torch
import argparse
import kornia
import imageio
import numpy as np

from train import HomographyModel
from dataset import SyntheticDataset, MEAN, STD
import os,glob
import os.path as osp

def tensors_to_gif(a, b, name):
    a = a.permute(1, 2, 0).numpy()
    b = b.permute(1, 2, 0).numpy()
    imageio.mimsave(name, [a, b], duration=1)


@torch.no_grad()
def main(args):
    # model = HomographyModel.load_from_checkpoint(args.checkpoint)
    model_dir = 'lightning_logs/version*'
    model_dir_list = sorted(glob.glob(model_dir))
    model_dir = model_dir_list[-1]
    model_path = osp.join(model_dir, "checkpoints", "*.ckpt")
    model_path_list = sorted(glob.glob(model_path))
    if len(model_path_list) > 0:
        model_path = model_path_list[-1]
        model = HomographyModel.load_from_checkpoint(model_path)
        print(model_path)
        print("model loaded.")
    else:
        raise ValueError(f'No load model!')  #raise Error
    model.eval()
    test_set = SyntheticDataset(args.test_path, rho=args.rho, filetype=args.filetype)

    #clear last output
    last_output = "figures/*"
    os.system("rm "+last_output)
    print('clear last ok.')
    for i in range(args.n):
        img_a, patch_a, patch_b, corners, delta = test_set[i]

        tensors_to_gif(patch_a, patch_b, f"figures/input_{i}.gif")
        patch_a = patch_a.unsqueeze(0)
        patch_b = patch_b.unsqueeze(0)
        corners = corners.unsqueeze(0)

        corners = corners - corners[:, 0].view(-1, 1, 2)

        delta_hat = model(patch_a, patch_b)
        corners_hat = corners + delta_hat
        h = kornia.get_perspective_transform(corners, corners_hat)
        h_inv = torch.inverse(h)

        patch_b_hat = kornia.warp_perspective(patch_a, h_inv, (128, 128))
        tensors_to_gif(patch_b_hat[0], patch_b[0], f"figures/output_{i}.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpoint", default="pretrained_coco.ckpt")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--rho", type=int, default=20, help="amount to perturb corners")
    parser.add_argument("--n", type=int, default=5, help="number of images to test")
    parser.add_argument("--filetype", default=".png")
    parser.add_argument("test_path", help="path to test images")
    args = parser.parse_args()
    main(args)
