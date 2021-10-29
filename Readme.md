# CompressAI
HESIC Project is inherited from https://github.com/InterDigitalInc/CompressAI

Installation：

```
pip install -e . 
pip install opencv-contrib-python==3.4.2.17 
pip install kornia 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda install pytorch==1.6.0 torchvision cudatoolkit=10.1
```



## test scripts:

cd /ywz/mywork/

```python test3real.py -d "/home/ywz/database/aftercut512"  --seed 0  --patch-size 512 512 --batch-size 1 --test-batch-size 1```

or

```python test3_savereal.py -d "/home/ywz/database/aftercut512"  --seed 0  --patch-size 512 512 --batch-size 1 --test-batch-size 1```



## datasets:

Pan Baidu :

link：https://pan.baidu.com/s/1sSbMCl-6LXPal_asBt5Giw 
code：k8rb 

Google Drive: link: https://drive.google.com/drive/folders/1tTMs8vf7Z4FAjwCg2aQVGA_pc9O_VpS1?usp=sharing



## pretrained_models:

Pan Baidu :

link：https://pan.baidu.com/s/1q0_2NZ46fYOCeDDg40nUaw 
code：qrfu 

Google Drive: link: https://drive.google.com/drive/folders/1tTMs8vf7Z4FAjwCg2aQVGA_pc9O_VpS1?usp=sharing



# Serialize

cd ywz/mywork

`newnet1.py` : HESIC 

`newnet1_joint.py` : HESIC+

`test2_codec.py` : test script for codec-compress & decompress  

​	-- import newnet1 or import newnet1_joint



cd ywz/DSIC

`mynet6_plus.py`: DSIC with codec

`mytrain2_test_codec.py`: test script for codec in DSIC




## Migration on Mindspore
https://github.com/ywz978020607/2021Summer-Image-Compression