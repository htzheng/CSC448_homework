# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
# ## GAP-TV for Video Compressive Sensing
# ### GAP-TV
# > X. Yuan, "Generalized alternating projection based total variation minimization for compressive sensing," in *IEEE International Conference on Image Processing (ICIP)*, 2016, pp. 2539-2543.
# ### Code credit
# [Xin Yuan](https://www.bell-labs.com/usr/x.yuan "Dr. Xin Yuan, Bell Labs"), [Bell Labs](https://www.bell-labs.com/), xyuan@bell-labs.com, created Aug 7, 2018.  
# [Yang Liu](https://liuyang12.github.io "Yang Liu, Tsinghua University"), [Tsinghua University](http://www.tsinghua.edu.cn/publish/thu2018en/index.html), y-liu16@mails.tsinghua.edu.cn, updated Jan 20, 2019.

# %%
import os
import time
import math
import h5py
import numpy as np
import scipy.io as sio
# import matplotlib.pyplot as plt
from statistics import mean

from ADMM_GAP_rec import (GAP_TV_rec, ADMM_TV_rec, GAP_FFDNet_rec, ADMM_FFDNet_rec)

from utils import (A_, At_)


# %%
# [0] environment configuration
datasetdir = './dataset/cacti' # dataset
# datasetdir = '../gapDenoise/dataset/cacti' # dataset
resultsdir = './results' # results

datname = 'kobe32'        # name of the dataset
# datname = 'traffic48'     # name of the dataset
# datname = 'runner40'      # name of the dataset
# datname = 'drop40'        # name of the dataset
# datname = 'crash32'       # name of the dataset
# datname = 'aerial32'      # name of the dataset
# datname = 'bicycle24'     # name of the dataset
# datname = 'starfish48'    # name of the dataset

# datname = 'starfish_c16_48'    # name of the dataset

matfile = datasetdir + '/' + datname + '_cacti.mat' # path of the .mat data file


# %%
from scipy.io.matlab.mio import _open_file
from scipy.io.matlab.miobase import get_matfile_version

# [1] load data
if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
    file = sio.loadmat(matfile) # for '-v7.2' and lower version of .mat file (MATLAB)
    order = 'K' # [order] keep as the default order in Python/numpy
    meas = np.float32(file['meas'])
    mask = np.float32(file['mask'])
    orig = np.float32(file['orig'])
else: # MATLAB .mat v7.3
    file =  h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
    order = 'F' # [order] switch to MATLAB array order
    meas = np.float32(file['meas'], order=order).transpose()
    mask = np.float32(file['mask'], order=order).transpose()
    orig = np.float32(file['orig'], order=order).transpose()

print(meas.shape, mask.shape, orig.shape)

iframe = 0
nframe = 1
# nframe = meas.shape[2]
MAXB = 255.

# common parameters and pre-calculation for PnP
# define forward model and its transpose
#A  = lambda x :  A_(x, mask) # forward model function handle
#At = lambda y : At_(y, mask) # transpose of forward model

mask_sum = np.sum(mask, axis=2)
mask_sum[mask_sum==0] = 1
[row,col,ColT] = mask.shape

# %%
## [2.3] GAP/ADMM-TV
### [2.3.1] GAP-TV
projmeth = 'gap' # projection method
_lambda = 1 # regularization factor
accelerate = True # enable accelerated version of GAP
denoiser = 'tv' # total variation (TV)
iter_max = 40 # maximum number of iterations
tv_weight = 0.3 # TV denoising weight (larger for smoother but slower)
tv_iter_max = 5 # TV denoising maximum number of iterations each
step_size = 1
eta = 1e-8

#ADMM_TV_rec(y,Phi,A, At,Phi_sum, maxiter, step_size, weight, row, col, ColT, eta,X_ori)
# only run the frist measurement
y  = meas[:,:,0]
X_ori  = orig[:,:,0:ColT]

# ADMM-TV
vadmm_tv = ADMM_TV_rec(y/255, mask, A_, At_,mask_sum, iter_max, step_size, tv_weight,row,col,ColT,eta, X_ori/255)

vgap_tv = GAP_TV_rec(y/255, mask, A_, At_,mask_sum, iter_max, step_size, tv_weight,row,col,ColT, X_ori/255)


# GAP-FFDnet
import torch
from packages.ffdnet.models import FFDNet

## [2.5] GAP/ADMM-FFDNet
### [2.5.1] GAP-FFDNet (FFDNet-based frame-wise video denoising)
projmeth = 'gap' # projection method
_lambda = 1 # regularization factor
accelerate = True # enable accelerated version of GAP
denoiser = 'ffdnet' # video non-local network 
noise_estimate = False # disable noise estimation for GAP

iter_max = 40 # maximum number of iterations
iter_max0 =  list(range(1, iter_max+1))
sigma    = [60/255*pow(0.971, k-1) for k in iter_max0]  # pre-set noise standard deviation
# sigma    = [12/255, 6/255] # pre-set noise standard deviation
# iter_max = [10,10] # maximum number of iterations
useGPU = True # use GPU

# pre-load the model for FFDNet image denoising
in_ch = 1
model_fn = 'packages/ffdnet/models/net_gray.pth'
# Absolute path to model file
# model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_fn)

# Create model
net = FFDNet(num_input_channels=in_ch)
# Load saved weights
if useGPU:
    state_dict = torch.load(model_fn)
    device_ids = [0]
    model = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
else:
    state_dict = torch.load(model_fn, map_location='cpu')
    # CPU mode: remove the DataParallel wrapper
    state_dict = remove_dataparallel_wrapper(state_dict)
    model = net
model.load_state_dict(state_dict)
model.eval() # evaluation mode

#GAP_FFDNet_rec(y,Phi,A, At,Phi_sum, maxiter, step_size,  row, col, ColT,nsig,model, X_ori)

# GAP_FFDnet
vgap_FFD = GAP_FFDNet_rec(y/255, mask, A_, At_,mask_sum, iter_max, step_size, row,col,ColT,sigma, model, X_ori/255)



# ADMM-FFDNet
vADMM_FFD = ADMM_FFDNet_rec(y/255,mask,A_, At_,mask_sum, iter_max, step_size, row, col, ColT, eta,sigma,model, X_ori/255)




