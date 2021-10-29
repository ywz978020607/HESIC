import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
import os,glob
import os.path as osp
import argparse
import pytorch_lightning as pl

from dataset import SyntheticDataset, safe_collate
from model import Net, photometric_loss


class HomographyModel(pl.LightningModule):
    def __init__(self, hparams):
        super(HomographyModel, self).__init__()
        self.hparams = hparams
        self.model = Net(patch_size=self.hparams.patchsize)

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch, batch_idx):
        img_a,img_b, patch_a, patch_b, corners, gt = batch   #非监督学习，不需要gt
        delta = self.model(patch_a, patch_b)  #两个patch输入网络，获得delta
        loss = photometric_loss(delta, img_a, patch_b, corners)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        img_a,img_b, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = photometric_loss(delta, img_a, patch_b, corners) #问题出在这里！ 换掉patch_b
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        train_set = SyntheticDataset(self.hparams.train_path, rho=self.hparams.rho,pic_size=self.hparams.picsize,patch_size=self.hparams.patchsize)
        return DataLoader(
            train_set,
            num_workers=4,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=safe_collate,
        )

    def val_dataloader(self):
        val_set = SyntheticDataset(self.hparams.valid_path, rho=self.hparams.rho,pic_size=self.hparams.picsize,patch_size=self.hparams.patchsize)
        return DataLoader(
            val_set,
            num_workers=4,
            batch_size=self.hparams.batch_size,
            collate_fn=safe_collate,
        )


def main(args):
    # if args.resume is not "":
    #     model = HomographyModel.load_from_checkpoint(args.resume)
    if args.resume == "none":
        model = HomographyModel(hparams=args)
        print("None:train from none.")
    elif args.resume is not "":
        model = HomographyModel(hparams=args)
        model_old = torch.load(args.resume, map_location=lambda storage, loc: storage)
        # print(model_old.keys())
        # net.load_state_dict(torch.load('path/params.pkl'))
        model.load_state_dict(model_old['state_dict'])
        # model = HomographyModel.load_from_checkpoint(args.resume)
        print(args.resume)
        print("model loaded.")
    else:
        try:
            model_dir = 'lightning_logs/version*'
            model_dir_list = sorted(glob.glob(model_dir))
            model_dir = model_dir_list[-1]
            model_path = osp.join(model_dir, "checkpoints", "*.ckpt")
            model_path_list = sorted(glob.glob(model_path))

            model_path = model_path_list[-1]
            model = HomographyModel.load_from_checkpoint(model_path)
            print(model_path)
            print("model loaded.")
        except:
            model = HomographyModel(hparams=args)
            print("train from none.")
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus)
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="batch size") #默认128
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--rho", type=int, default=45, help="amount to perturb corners")

    parser.add_argument("--picsize", type=int, default=512)
    parser.add_argument("--patchsize", type=int, default=256)

    parser.add_argument(
        "--resume", type=str, help="checkpoint to resume from", default=""
    )
    parser.add_argument("train_path", help="path to training imgs")
    parser.add_argument("valid_path", help="path to validation imgs")
    args = parser.parse_args()
    main(args)
