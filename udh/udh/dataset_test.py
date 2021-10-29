# 最新状态：no use ready to delete
# for test 多返回一个img_b
import torch
import random
from pathlib import Path
import kornia
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

MEAN = torch.tensor([0.485, 0.456, 0.406]).mean().unsqueeze(0)
STD = torch.tensor([0.229, 0.224, 0.225]).mean().unsqueeze(0)


def safe_collate(batch):
    """Return batch without any None values"""
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class SyntheticDataset(Dataset):
    def __init__(self, folder, filetype=".png", pic_size = 256,patch_size=128, rho=45):
        super(SyntheticDataset, self).__init__()
        self.fnames = list((Path(folder)/'left').glob(f"*{filetype}"))
        self.transforms = transforms.Compose(
            [
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                #ywz
                # transforms.Resize(pic_size),
                # #
                # transforms.CenterCrop(pic_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
        self.pic_size = pic_size
        self.patch_size = patch_size
        self.rho = rho

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # print(self.fnames[index])
        img_a = Image.open(self.fnames[index])
        img_a = self.transforms(img_a)  #[3,H,W]
        # grayscale
        img_a = torch.mean(img_a, dim=0, keepdim=True)  #[1,H,W]
        # print(img_a.size())
        # print((str)(self.fnames[index]))
        # ywz load img_b
        b_path = (str)(self.fnames[index]).replace("left", "right")
        # print(b_path)
        img_b = Image.open(b_path)
        img_b = self.transforms(img_b)
        # grayscale
        img_b = torch.mean(img_b, dim=0, keepdim=True)
        ###

        # pick top left corner
        x = random.randint(self.rho, self.pic_size - self.rho - self.patch_size)
        y = random.randint(self.rho, self.pic_size - self.rho - self.patch_size)

        # x = 0
        # y = 0

        corners = torch.tensor(
            [
                [x, y],
                [x + self.patch_size, y],
                [x + self.patch_size, y + self.patch_size],
                [x, y + self.patch_size],
            ]
        )
        delta = torch.randint_like(corners, -self.rho, self.rho)
        perturbed_corners = corners + delta

        # try:
        #     # compute homography from points
        #     h = kornia.get_perspective_transform(
        #         corners.unsqueeze(0).float(), perturbed_corners.unsqueeze(0).float()
        #     )
        #
        #     h_inv = torch.inverse(h)
        #
        #     # apply homography to single img
        #     img_b = kornia.warp_perspective(img_a.unsqueeze(0), h_inv, (self.pic_size, self.pic_size))[0]
        # except:
        #     # either matrix could not be solved or inverted
        #     # this will show up as None, so use safe_collate in train.py
        #     return
        img_a.unsqueeze(0)
        patch_a = img_a[:, y : y + self.patch_size, x : x + self.patch_size]
        patch_b = img_b[:, y : y + self.patch_size, x : x + self.patch_size]

        return img_a,img_b, patch_a, patch_b, corners.float(), delta.float()   #最后一项用于监督学习，在非监督学习中用不到
