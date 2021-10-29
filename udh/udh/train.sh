# 打印时间 在随后加上  && echo $(date +%Y/%m/%d/%R)

#train
python train.py /home/sharklet/database/aftercut/train/ /home/sharklet/database/aftercut/test/

python train.py /home/sharklet/database/aftercut512/train/ /home/sharklet/database/aftercut512/test/ --picsize 512 --patchsize 128 --gpus "0"  --resume "lightning_logs/512_128_version_18/checkpoints/epoch=62.ckpt"
python train.py /home/sharklet/database/aftercut512/train/ /home/sharklet/database/aftercut512/test/ --picsize 512 --patchsize 256 --gpus "3"  && echo $(date +%Y/%m/%d/%R)
python train.py /home/ywz/database/aftercut512/train/ /home/ywz/database/aftercut512/test/ --picsize 512 --patchsize 256 --gpus "3"  && echo $(date +%Y/%m/%d/%R)
python train.py /home/sharklet/database/aftercut/train/ /home/sharklet/database/aftercut/test/ --picsize 256 --patchsize 128 --resume "pretrained_coco.ckpt"




#test
python test3.py /home/ywz/database/aftercut512/test/ --resume "lightning_logs/512_128_version_18/checkpoints/epoch=62.ckpt"
python test3.py /home/ywz/database/aftercut512/test/

python QHtest.py /home/ywz/database/aftercut512/test/

