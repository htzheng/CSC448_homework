import time
import math
import skimage
import numpy as np
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
# from packages.vnlnet.test import vnlnet
from packages.ffdnet.test_ffdnet_ipol import ffdnet_vdenoiser
from packages.fastdvdnet.test_fastdvdnet import fastdvdnet_denoiser
from utils import (A_, At_, psnr)
if skimage.__version__ < '0.18':
    from skimage.measure import (compare_psnr, compare_ssim)
else: # skimage.measure deprecated in version 0.18 ( -> skimage.metrics )
    import skimage.metrics.peak_signal_noise_ratio as compare_psnr
    import skimage.metrics.structural_similarity   as compare_ssim

#@author: Xin


def GAP_TV_rec(y,Phi,A, At,Phi_sum, maxiter, step_size, weight, row, col, ColT, X_ori):
    y1 = np.zeros((row,col))
    begin_time = time.time()
    f = At(y,Phi)
    for ni in range(maxiter):
        fb = A(f,Phi)
        y1 = y1+ (y-fb)
        f  = f + np.multiply(step_size, At( np.divide(y1-fb,Phi_sum),Phi ))
        f = denoise_tv_chambolle(f, weight,n_iter_max=30,multichannel=True)
    
        if (ni+1)%5 == 0:
            # mse = np.mean(np.sum((y-A(f,Phi))**2,axis=(0,1)))
            end_time = time.time()
            print("GAP-TV: Iteration %3d, PSNR = %2.2f dB,"
              " time = %3.1fs."
              % (ni+1, psnr(f, X_ori), end_time-begin_time))
    return f

def ADMM_TV_rec(y,Phi,A, At,Phi_sum, maxiter, step_size, weight, row, col, ColT, eta,X_ori):
    #y1 = np.zeros((row,col))
    begin_time = time.time()
    theta = At(y,Phi)
    v =theta
    b = np.zeros((row,col,ColT))
    for ni in range(maxiter):
        yb = A(theta+b,Phi)
        #y1 = y1+ (y-fb)
        v  = (theta+b) + np.multiply(step_size, At( np.divide(y-yb,Phi_sum+eta),Phi ))
        #vmb = v-b
        theta = denoise_tv_chambolle(v-b, weight,n_iter_max=30,multichannel=True)
        
        b = b-(v-theta)
        weight = 0.999*weight
        eta = 0.998 * eta
        
        if (ni+1)%5 == 0:
            # mse = np.mean(np.sum((y-A(v,Phi))**2,axis=(0,1)))
            end_time = time.time()
            print("ADMM-TV: Iteration %3d, PSNR = %2.2f dB,"
              " time = %3.1fs."
              % (ni+1, psnr(v, X_ori), end_time-begin_time))
    return v



def GAP_FFDNet_rec(y,Phi,A, At,Phi_sum, maxiter, step_size,  row, col, ColT,nsig,model, X_ori):
    y1 = np.zeros((row,col))
    begin_time = time.time()
    f = At(y,Phi)
    for ni in range(maxiter):
        fb = A(f,Phi)
        y1 = y1+ (y-fb)
        f  = f + np.multiply(step_size, At( np.divide(y1-fb,Phi_sum),Phi ))
        #f = denoise_tv_chambolle(f, weight,n_iter_max=30,multichannel=True)
        f = ffdnet_vdenoiser(f, nsig[ni], model)
    
        if (ni+1)%5 == 0:
            # mse = np.mean(np.sum((y-A(f,Phi))**2,axis=(0,1)))
            end_time = time.time()
            print("GAP-TV: Iteration %3d, PSNR = %2.2f dB,"
              " time = %3.1fs."
              % (ni+1, psnr(f, X_ori), end_time-begin_time))
    return f

def ADMM_FFDNet_rec(y,Phi,A, At,Phi_sum, maxiter, step_size, row, col, ColT, eta,nsig,model, X_ori):
    #y1 = np.zeros((row,col))
    begin_time = time.time()
    theta = At(y,Phi)
    v =theta
    b = np.zeros((row,col,ColT))
    for ni in range(maxiter):
        yb = A(theta+b,Phi)
        #y1 = y1+ (y-fb)
        v  = (theta+b) + np.multiply(step_size, At( np.divide(y-yb,Phi_sum+eta),Phi ))
        #vmb = v-b
        #theta = denoise_tv_chambolle(v-b, weight,n_iter_max=30,multichannel=True)
        theta = ffdnet_vdenoiser(v-b, nsig[ni], model)
        
        b = b-(v-theta)
        # weight = 0.999*weight
        eta = 0.998 * eta
        
        if (ni+1)%5 == 0:
            # mse = np.mean(np.sum((y-A(v,Phi))**2,axis=(0,1)))
            end_time = time.time()
            print("ADMM-TV: Iteration %3d, PSNR = %2.2f dB,"
              " time = %3.1fs."
              % (ni+1, psnr(v, X_ori), end_time-begin_time))
    return v

import torch


def ADMM_TV_DIP(x0, y, maxiter, denoise_weight, model, X_ori):
    b = torch.zeros(x0)
    begin_time = time.time()

    for ni in range(maxiter):
        model.train(x0, y, b) # update transfer network parameters theta
        x = denoise_tv_chambolle(model.eval(x0, y)-b, denoise_weight, n_iter_max=30, multichannel=True) # update x
        b = b - (x - model.eval(x0, y)) # update b

        if (ni + 1) % 5 == 0:

            end_time = time.time()
            print("ADMM-FFDNet-DIP: Iteration %3d, PSNR = %2.2f dB,"
                  " time = %3.1fs."
                  % (ni + 1, psnr(x, X_ori), end_time - begin_time))
    return x

def ADMM_FFDNet_DIP(x0, y, maxiter, model, X_ori):
    b = torch.zeros(x0)
    begin_time = time.time()

    for ni in range(maxiter):
        model.train(x0, y, b) # update transfer network parameters theta
        x = ffdnet_vdenoiser(model.eval(x0, y)-b) # update x
        b = b - (x - model.eval(x0, y)) # update b

        if (ni + 1) % 5 == 0:

            end_time = time.time()
            print("ADMM-FFDNet-DIP: Iteration %3d, PSNR = %2.2f dB,"
                  " time = %3.1fs."
                  % (ni + 1, psnr(x, X_ori), end_time - begin_time))
    return x

