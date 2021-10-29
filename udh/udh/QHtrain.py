import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
import os, glob
import os.path as osp
import argparse
import torch.nn as nn
import shutil

from dataset import SyntheticDataset, safe_collate
from model import Net, photometric_loss


class HomographyModel(nn.Module):
    def __init__(self, hparams):
        super(HomographyModel, self).__init__()
        self.hparams = hparams
        self.model = Net(patch_size=self.hparams.patchsize)

    def forward(self, a, b):
        return self.model(a, b)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoint_best_loss.pth.tar')


def main(args):
    # if args.resume is not "":
    #     model = HomographyModel.load_from_checkpoint(args.resume)
    if args.resume == "none":
        hmodel = HomographyModel(hparams=args)
        best = 0
        print("None:train from none.")
    elif args.resume is not "":
        hmodel = HomographyModel(hparams=args)
        model_old = torch.load(args.resume, map_location=lambda storage, loc: storage)
        # print(model_old.keys())
        # net.load_state_dict(torch.load('path/params.pkl'))
        hmodel.load_state_dict(model_old['state_dict'])
        best = model_old['loss']
        # model = HomographyModel.load_from_checkpoint(args.resume)
        print(args.resume)
        print("model loaded.")
    else:
        try:
            model_dir = 'HESIC_logs/version*'
            model_dir_list = sorted(glob.glob(model_dir))
            model_dir = model_dir_list[-1]
            model_path = osp.join(model_dir, "checkpoints", "*.ckpt")
            model_path_list = sorted(glob.glob(model_path))

            model_path = model_path_list[-1]
            hmodel = HomographyModel.load_from_checkpoint(model_path)
            best = hmodel['loss']
            print(model_path)
            print("model loaded.")
        except:
            hmodel = HomographyModel(hparams=args)
            best = 0
            print("train from none.")
    print("history_best:", best)

    device = torch.device("cuda:"+str(args.gpus) if torch.cuda.is_available() else "cpu")
    hmodel = hmodel.to(device)

    train_set = SyntheticDataset(hmodel.hparams.train_path, rho=hmodel.hparams.rho, pic_size=hmodel.hparams.picsize,
                                 patch_size=hmodel.hparams.patchsize)
    val_set = SyntheticDataset(hmodel.hparams.valid_path, rho=hmodel.hparams.rho, pic_size=hmodel.hparams.picsize,
                               patch_size=hmodel.hparams.patchsize)
    train_loader = DataLoader(
        train_set,
        num_workers=4,
        batch_size=hmodel.hparams.batch_size,
        shuffle=True,
        collate_fn=safe_collate,
    )
    validation_loader = DataLoader(
            val_set,
            num_workers=4,
            batch_size=hmodel.hparams.batch_size,
            collate_fn=safe_collate,
        )

    optimizer = torch.optim.Adam(hmodel.model.parameters(), lr=hmodel.hparams.learning_rate)

    for epoch in range(args.epochs):
        for train_step, train_batch in enumerate(train_loader):
            img_a, img_b, patch_a, patch_b, corners, gt = train_batch
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            patch_a = patch_a.to(device)
            patch_b = patch_b.to(device)
            corners = corners.to(device)
            delta = hmodel.model(patch_a, patch_b).to(device)
            optimizer.zero_grad()
            loss = photometric_loss(delta, img_a, patch_b, corners)
            loss.backward()
            optimizer.step()
            if train_step % 100 == 0:
                print("train || epoch:", epoch, "  step:", train_step, "  loss:", loss.data)

        with torch.no_grad():
            vali_total_loss = []
            for vali_step, validation_batch in enumerate(validation_loader):
                vali_img_a, vali_img_b, vali_patch_a, vali_patch_b, vali_corners, vali_gt = validation_batch
                vali_img_a = vali_img_a.to(device)
                vali_img_b = vali_img_b.to(device)
                vali_patch_a = vali_patch_a.to(device)
                vali_patch_b = vali_patch_b.to(device)
                vali_corners = vali_corners.to(device)
                vali_delta = hmodel.model(vali_patch_a, vali_patch_b).to(device)
                vali_loss = photometric_loss(vali_delta, vali_img_a, vali_patch_b, vali_corners)
                vali_total_loss.append({"vali_loss": vali_loss})

            avg_loss = torch.stack([x["vali_loss"] for x in vali_total_loss]).mean()
            print("test || loss:", avg_loss.data)
            if best == 0:
                best = avg_loss
                is_best = 1
            elif avg_loss <= best:
                is_best = 1
                best = avg_loss
            else:
                is_best = 0
            save_checkpoint({
                        'state_dict': hmodel.state_dict(),
                        'loss': avg_loss,
                        'optimizer': optimizer.state_dict(),
                    }, is_best)
            print("present best = ", best)
            torch.cuda.empty_cache()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="batch size") #默认128
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--rho", type=int, default=45, help="amount to perturb corners")

    parser.add_argument("--picsize", type=int, default=256)
    parser.add_argument("--patchsize", type=int, default=128)

    parser.add_argument(
        "--resume", type=str, help="checkpoint to resume from", default=""
    )
    parser.add_argument("train_path", help="path to training imgs")
    parser.add_argument("valid_path", help="path to validation imgs")
    args = parser.parse_args()
    main(args)
