#!/usr/bin/env python
# coding: utf-8

# <!-- Copyright 2020 InterDigital Communications, Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# # Compressai Models Comparison Demo

# In[1]:


import math
import io
import torch
from torchvision import transforms
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt


# In[2]:


from pytorch_msssim import ms_ssim


# In[3]:


from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean)


# In[4]:


from ipywidgets import interact, widgets


# ## Global settings

# In[1]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
metric = 'mse'  # only pre-trained model for mse are available for now
quality = 1     # lower quality -> lower bit-rate (use lower quality to clearly see visual differences in the notebook)


# ## Load some pretrained models

# In[2]:


networks = {
    'bmshj2018-factorized': bmshj2018_factorized(quality=quality, pretrained=True).eval().to(device),
    'bmshj2018-hyperprior': bmshj2018_hyperprior(quality=quality, pretrained=True).eval().to(device),
    'mbt2018-mean': mbt2018_mean(quality=quality, pretrained=True).eval().to(device)
}


# ## Inference

# ### Load input data

# In[7]:


img = Image.oimg = Image.open('./assets/stmalo_fracape.png').convert('RGB')
x = transforms.ToTensor()(img).unsqueeze(0)


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(9, 6))
plt.axis('off')
plt.imshow(img)
plt.show()


# ### Run the networks

# In[9]:


outputs = {}
with torch.no_grad():
    for name, net in networks.items():
        rv = net(x)
        rv['x_hat'].clamp_(0, 1)
        outputs[name] = rv


# ### Visualize the reconstructions

# In[4]:


test_ywz = "123"
print(test_ywz)


# In[10]:


reconstructions = {name: transforms.ToPILImage()(out['x_hat'].squeeze())
                  for name, out in outputs.items()}


# In[11]:


diffs = [torch.mean((out['x_hat'] - x).abs(), axis=1).squeeze()
        for out in outputs.values()]


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
fix, axes = plt.subplots(1, 3, figsize=(16, 12))
for ax in axes.ravel():
    ax.axis('off')
    
for i, (name, rec) in enumerate(reconstructions.items()):
    axes[i].imshow(rec.crop((468, 212, 768, 512))) # cropped for easy comparison
    axes[i].title.set_text(name)

plt.show()


# ## Metric

# In[13]:


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()


# In[14]:


metrics = {}
for name, out in outputs.items():
    metrics[name] = {
        'psnr': compute_psnr(x, out["x_hat"]),
        'ms-ssim': compute_msssim(x, out["x_hat"]),
        'bit-rate': compute_bpp(out),
    }


# In[15]:


header = f'{"Model":20s} | {"PSNR [dB]"} | {"MS-SSIM":<9s} | {"Bpp":<9s}|'
print('-'*len(header))
print(header)
print('-'*len(header))
for name, m in metrics.items():
    print(f'{name:20s}', end='')
    for v in m.values():
        print(f' | {v:9.3f}', end='')
    print('|')
print('-'*len(header))


# In[16]:


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plt.figtext(.5, 0., '(upper-left is better)', fontsize=12, ha='center')
for name, m in metrics.items():
    axes[0].plot(m['bit-rate'], m['psnr'], 'o', label=name)
    axes[0].legend(loc='best')
    axes[0].grid()
    axes[0].set_ylabel('PSNR [dB]')
    axes[0].set_xlabel('Bit-rate [bpp]')
    axes[0].title.set_text('PSNR comparison')

    axes[1].plot(m['bit-rate'], -10*np.log10(1-m['ms-ssim']), 'o', label=name)
    axes[1].legend(loc='best')
    axes[1].grid()
    axes[1].set_ylabel('MS-SSIM [dB]')
    axes[1].set_xlabel('Bit-rate [bpp]')
    axes[1].title.set_text('MS-SSIM (log) comparison')

plt.show()

