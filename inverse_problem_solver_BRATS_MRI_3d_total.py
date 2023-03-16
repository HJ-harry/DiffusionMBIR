from pathlib import Path
from models import utils as mutils
import sampling
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      LangevinCorrectorCS)
from models import ncsnpp
from itertools import islice
from losses import get_optimizer
import datasets
import time
import controllable_generation_TV
from utils import restore_checkpoint, fft2, ifft2, show_samples_gray, get_mask, clear
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
from scipy.io import savemat, loadmat
from tqdm import tqdm
import matplotlib.pyplot as plt
import importlib


###############################################
# Configurations
###############################################
problem = 'Fourier_CS_3d_admm_tv'
config_name = 'fastmri_knee_320_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 2000
ckpt_num = 95
N = num_scales

root = './data/MRI/BRATS'
vol = 'Brats18_CBICA_AAM_1'

if sde.lower() == 'vesde':
  # from configs.ve import fastmri_knee_320_ncsnpp_continuous as configs
  configs = importlib.import_module(f"configs.ve.{config_name}")
  if config_name == 'fastmri_knee_320_ncsnpp_continuous':
    ckpt_filename = f"./exp/ve/{config_name}/checkpoint_{ckpt_num}.pth"
  elif config_name == 'ffhq_256_ncsnpp_continuous':
    ckpt_filename = f"exp/ve/{config_name}/checkpoint_48.pth"
  config = configs.get_config()
  config.model.num_scales = num_scales
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  sde.N = N
  sampling_eps = 1e-5

img_size = 240
batch_size = 1
config.training.batch_size = batch_size
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
probability_flow = False
snr = 0.16
n_steps = 1

# parameters for Fourier CS recon
mask_type = 'uniform1d'
use_measurement_noise = False
acc_factor = 2.0
center_fraction = 0.15

# ADMM TV parameters
lamb_list = [0.005]
rho_list = [0.01]

random_seed = 0

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)
state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True)
ema.copy_to(score_model.parameters())

fname_list = sorted(list((Path(root) / vol).glob('*.npy')))
all_img = []
for fname in tqdm(fname_list):
    img = np.load(fname)
    img = torch.from_numpy(img)
    h, w = img.shape
    img = img.view(1, 1, h, w)
    all_img.append(img)

all_img = torch.cat(all_img, dim=0)

# normalize the volume to be in proper range
vmax = all_img.max()
all_img /= (vmax + 1e-5)

img = all_img.to(config.device)
b = img.shape[0]

for lamb in lamb_list:
    for rho in rho_list:
        print(f'lambda: {lamb}')
        print(f'rho:    {rho}')
        # Specify save directory for saving generated samples
        save_root = Path(f'./results/{config_name}/{problem}/{mask_type}/acc{acc_factor}/lamb{lamb}/rho{rho}/{vol}')
        save_root.mkdir(parents=True, exist_ok=True)

        irl_types = ['input', 'recon', 'label']
        for t in irl_types:
            save_root_f = save_root / t
            save_root_f.mkdir(parents=True, exist_ok=True)

        ###############################################
        # Inference
        ###############################################

        # forward model
        kspace = fft2(img)

        # generate mask
        mask = get_mask(torch.zeros(1, 1, h, w), img_size, batch_size,
                        type=mask_type, acc_factor=acc_factor, center_fraction=center_fraction)
        mask = mask.to(img.device)
        mask = mask.repeat(b, 1, 1, 1)

        pc_fouriercs = controllable_generation_TV.get_pc_radon_ADMM_TV_mri(sde,
                                                                           predictor, corrector,
                                                                           inverse_scaler,
                                                                           mask=mask,
                                                                           lamb_1=lamb,
                                                                           rho=rho,
                                                                           img_shape=img.shape,
                                                                           snr=snr,
                                                                           n_steps=n_steps,
                                                                           probability_flow=probability_flow,
                                                                           continuous=config.training.continuous)

        # undersampling
        under_kspace = kspace * mask
        under_img = torch.real(ifft2(under_kspace))

        count = 0
        for i, recon_img in enumerate(under_img):
            plt.imsave(save_root / 'input' / f'{count}.png', clear(under_img[i]), cmap='gray')
            plt.imsave(save_root / 'label' / f'{count}.png', clear(img[i]), cmap='gray')
            count += 1

        x = pc_fouriercs(score_model, scaler(under_img), measurement=under_kspace)

        count = 0
        for i, recon_img in enumerate(x):
            plt.imsave(save_root / 'input' / f'{count}.png', clear(under_img[i]), cmap='gray')
            plt.imsave(save_root / 'label' / f'{count}.png', clear(img[i]), cmap='gray')
            plt.imsave(save_root / 'recon' / f'{count}.png', clear(recon_img), cmap='gray')
            np.save(str(save_root / 'input' / f'{count}.npy'), clear(under_img[i], normalize=False))
            np.save(str(save_root / 'recon' / f'{count}.npy'), clear(x[i], normalize=False))
            np.save(str(save_root / 'label' / f'{count}.npy'), clear(img[i], normalize=False))
            count += 1

