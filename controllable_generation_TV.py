import functools
import time

import torch
from numpy.testing._private.utils import measure
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import utils as mutils
from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
from utils import fft2, ifft2, fft2_m, ifft2_m
from physics.ct import *
from utils import show_samples, show_samples_gray, clear, clear_color, batchfy



class lambda_schedule:
  def __init__(self, total=2000):
    self.total = total

  def get_current_lambda(self, i):
    pass
class lambda_schedule_linear(lambda_schedule):
  def __init__(self, start_lamb=1.0, end_lamb=0.0):
    super().__init__()
    self.start_lamb = start_lamb
    self.end_lamb = end_lamb

  def get_current_lambda(self, i):
    return self.start_lamb + (self.end_lamb - self.start_lamb) * (i / self.total)


class lambda_schedule_const(lambda_schedule):
  def __init__(self, lamb=1.0):
    super().__init__()
    self.lamb = lamb

  def get_current_lambda(self, i):
    return self.lamb


def _Dz(x): # Batch direction
    y = torch.zeros_like(x)
    y[:-1] = x[1:]
    y[-1] = x[0]
    return y - x


def _DzT(x): # Batch direction
    y = torch.zeros_like(x)
    y[:-1] = x[1:]
    y[-1] = x[0]

    tempt = -(y-x)
    difft = tempt[:-1]
    y[1:] = difft
    y[0] = x[-1] - x[0]

    return y

def _Dx(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :-1, :] = x[:, :, 1:, :]
    y[:, :, -1, :] = x[:, :, 0, :]
    return y - x


def _DxT(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :-1, :] = x[:, :, 1:, :]
    y[:, :, -1, :] = x[:, :, 0, :]
    tempt = -(y - x)
    difft = tempt[:, :, :-1, :]
    y[:, :, 1:, :] = difft
    y[:, :, 0, :] = x[:, :, -1, :] - x[:, :, 0, :]
    return y


def _Dy(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :, :-1] = x[:, :, :, 1:]
    y[:, :, :, -1] = x[:, :, :, 0]
    return y - x


def _DyT(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :, :-1] = x[:, :, :, 1:]
    y[:, :, :, -1] = x[:, :, :, 0]
    tempt = -(y - x)
    difft = tempt[:, :, :, :-1]
    y[:, :, :, 1:] = difft
    y[:, :, :, 0] = x[:, :, :, -1] - x[:, :, :, 0]
    return y


def get_pc_radon_ADMM_TV(sde, predictor, corrector, inverse_scaler, snr,
                         n_steps=1, probability_flow=False, continuous=False,
                         denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None,
                         final_consistency=False, img_cache=None, img_shape=None, lamb_1=5, rho=10):
    """ Sparse application of measurement consistency """
    # Define predictor & corrector
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    if img_cache != None :
        img_shape[0] += 1
    del_z = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def kaczmarz(x, x_mean, measurement=None, lamb=1.0, i=None,
                 norm_const=None):
        x = x + lamb * _AT(measurement - _A(x))/norm_const
        x_mean = x
        return x, x_mean
    
    def A_cg(x):
        return _AT(_A(x)) + rho * _DzT(_Dz(x))

    def CG(A_fn,b_cg,x,n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1,-1),r.view(1,-1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old/torch.matmul(p.view(1,-1),Ap.view(1,-1).T)
    
            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1,-1),r.view(1,-1).T)
            if torch.sqrt(rs_new) < eps :
                break
            p = r + (rs_new/rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(x,ATy, niter=20):
        if img_cache != None :
            x = torch.cat([img_cache,x],dim=0)
            idx = list(range(len(x),0,-1))
            x = x[idx]

        nonlocal del_z, udel_z
        if del_z.device != x.device :
            del_z = del_z.to(x.device)
            udel_z = del_z.to(x.device)
        for i in range(niter):
            b_cg = ATy + rho * (_DzT(del_z)-_DzT(udel_z))
            x = CG(A_cg, b_cg, x, n_inner=1)

            del_z = shrink(_Dz(x) + udel_z, lamb_1/rho)
            udel_z = _Dz(x) - del_z + udel_z
        if img_cache != None :
            x = x[idx]
            x = x[1:]
            del_z[-1] = 0
            udel_z[-1] = 0
        x_mean = x
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean
        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, x, t, measurement=None):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=1)
                return x, x_mean
        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)

            ones = torch.ones_like(x).to(data.device)
            norm_const = _AT(_A(ones))
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                x, x_mean = predictor_denoise_update_fn(model, data, x, t)
                x, x_mean = corrector_radon_update_fn(model, data, x, t, measurement=measurement)
                if save_progress:
                    if (i % 50) == 0:
                        print(f'iter: {i}/{sde.N}')
                        plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x_mean[0:1]), cmap='gray')
            # Final step which coerces the data fidelity error term to be zero,
            # and thereby satisfying Ax = y
            if final_consistency:
                x, x_mean = kaczmarz(x, x_mean, measurement, lamb=1.0, norm_const=norm_const)

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon


def get_pc_radon_ADMM_TV_vol(sde, predictor, corrector, inverse_scaler, snr,
                             n_steps=1, probability_flow=False, continuous=False,
                             denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None,
                             final_consistency=False, img_shape=None, lamb_1=5, rho=10):
    """ Sparse application of measurement consistency """
    # Define predictor & corrector
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    del_z = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def kaczmarz(x, x_mean, measurement=None, lamb=1.0, i=None,
                 norm_const=None):
        x = x + lamb * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean

    def A_cg(x):
        return _AT(_A(x)) + rho * _DzT(_Dz(x))

    def CG(A_fn, b_cg, x, n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(x, ATy, niter=20):
        nonlocal del_z, udel_z
        if del_z.device != x.device:
            del_z = del_z.to(x.device)
            udel_z = del_z.to(x.device)
        for i in range(niter):
            b_cg = ATy + rho * (_DzT(del_z) - _DzT(udel_z))
            x = CG(A_cg, b_cg, x, n_inner=1)

            del_z = shrink(_Dz(x) + udel_z, lamb_1 / rho)
            udel_z = _Dz(x) - del_z + udel_z
        x_mean = x
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=1)
                return x, x_mean
        return ADMM_TV_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    mc_update_fn = get_ADMM_TV_fn()

    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)

            ones = torch.ones_like(x).to(data.device)
            norm_const = _AT(_A(ones))
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                # 1. batchify into sizes that fit into the GPU
                x_batch = batchfy(x, 12)
                # 2. Run PC step for each batch
                x_agg = list()
                for idx, x_batch_sing in enumerate(x_batch):
                    x_batch_sing, _ = predictor_denoise_update_fn(model, data, x_batch_sing, t)
                    x_batch_sing, _ = corrector_denoise_update_fn(model, data, x_batch_sing, t)
                    x_agg.append(x_batch_sing)
                # 3. Aggregate to run ADMM TV
                x = torch.cat(x_agg, dim=0)
                # 4. Run ADMM TV
                x, x_mean = mc_update_fn(x, measurement=measurement)

                if save_progress:
                    if (i % 50) == 0:
                        print(f'iter: {i}/{sde.N}')
                        plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear(x_mean[0:1]), cmap='gray')
            # Final step which coerces the data fidelity error term to be zero,
            # and thereby satisfying Ax = y
            if final_consistency:
                x, x_mean = kaczmarz(x, x, measurement, lamb=1.0, norm_const=norm_const)

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon


def get_pc_radon_ADMM_TV_all_vol(sde, predictor, corrector, inverse_scaler, snr,
                             n_steps=1, probability_flow=False, continuous=False,
                             denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None,
                             final_consistency=False, img_shape=None, lamb_1=5, rho=10):
    """ Sparse application of measurement consistency """
    # Define predictor & corrector
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    del_x = torch.zeros(img_shape)
    del_y = torch.zeros(img_shape)
    del_z = torch.zeros(img_shape)
    udel_x = torch.zeros(img_shape)
    udel_y = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def kaczmarz(x, x_mean, measurement=None, lamb=1.0, i=None,
                 norm_const=None):
        x = x + lamb * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean


    def A_cg(x):
        return _AT(_A(x)) + rho * (_DxT(_Dx(x)) + _DyT(_Dy(x)) + _DzT(_Dz(x)))

    def CG(A_fn, b_cg, x, n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(x, ATy, niter=20):
        nonlocal del_x, del_y, del_z, udel_x, udel_y, udel_z
        if del_z.device != x.device:
            del_x = del_x.to(x.device)
            del_y = del_y.to(x.device)
            del_z = del_z.to(x.device)
            udel_x = udel_x.to(x.device)
            udel_y = udel_y.to(x.device)
            udel_z = udel_z.to(x.device)
        for i in range(niter):
            b_cg = ATy + rho * ((_DxT(del_x) - _DxT(udel_x))
                                + (_DyT(del_y) - _DyT(udel_y))
                                + (_DzT(del_z) - _DzT(udel_z)))
            x = CG(A_cg, b_cg, x, n_inner=1)

            del_x = shrink(_Dx(x) + udel_x, lamb_1 / rho)
            del_y = shrink(_Dy(x) + udel_y, lamb_1 / rho)
            del_z = shrink(_Dz(x) + udel_z, lamb_1 / rho)
            udel_x = _Dx(x) - del_x + udel_x
            udel_y = _Dy(x) - del_y + udel_y
            udel_z = _Dz(x) - del_z + udel_z
        x_mean = x
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=1)
                return x, x_mean
        return ADMM_TV_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    mc_update_fn = get_ADMM_TV_fn()

    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)

            ones = torch.ones_like(x).to(data.device)
            norm_const = _AT(_A(ones))
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                # 1. batchify into sizes that fit into the GPU
                x_batch = batchfy(x, 12)
                # 2. Run PC step for each batch
                x_agg = list()
                for idx, x_batch_sing in enumerate(x_batch):
                    x_batch_sing, _ = predictor_denoise_update_fn(model, data, x_batch_sing, t)
                    x_batch_sing, _ = corrector_denoise_update_fn(model, data, x_batch_sing, t)
                    x_agg.append(x_batch_sing)
                # 3. Aggregate to run ADMM TV
                x = torch.cat(x_agg, dim=0)
                # 4. Run ADMM TV
                x, x_mean = mc_update_fn(x, measurement=measurement)

                if save_progress:
                    if (i % 50) == 0:
                        print(f'iter: {i}/{sde.N}')
                        plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear(x_mean[0:1]), cmap='gray')
            # Final step which coerces the data fidelity error term to be zero,
            # and thereby satisfying Ax = y
            if final_consistency:
                x, x_mean = kaczmarz(x, x, measurement, lamb=1.0, norm_const=norm_const)

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon



def get_ADMM_TV(eps=1e-5, radon=None, save_progress=False, save_root=None,
                img_shape=None, lamb_1=5, rho=10, outer_iter=30, inner_iter=20):

    del_x = torch.zeros(img_shape)
    del_y = torch.zeros(img_shape)
    del_z = torch.zeros(img_shape)
    udel_x = torch.zeros(img_shape)
    udel_y = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def A_cg(x):
        return _AT(_A(x)) + rho * (_DxT(_Dx(x)) + _DyT(_Dy(x)) + _DzT(_Dz(x)))

    def CG(A_fn, b_cg, x, n_inner=20):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(x, ATy, niter=30):
        nonlocal del_x, del_y, del_z, udel_x, udel_y, udel_z
        if del_z.device != x.device:
            del_x = del_x.to(x.device)
            del_y = del_y.to(x.device)
            del_z = del_z.to(x.device)
            udel_x = udel_x.to(x.device)
            udel_y = udel_y.to(x.device)
            udel_z = udel_z.to(x.device)
        for i in tqdm(range(niter)):
            b_cg = ATy + rho * ((_DxT(del_x) - _DxT(udel_x))
                                + (_DyT(del_y) - _DyT(udel_y))
                                + (_DzT(del_z) - _DzT(udel_z)))
            x = CG(A_cg, b_cg, x, n_inner=inner_iter)
            if save_progress:
                plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear(x[0:1]), cmap='gray')

            del_x = shrink(_Dx(x) + udel_x, lamb_1 / rho)
            del_y = shrink(_Dy(x) + udel_y, lamb_1 / rho)
            del_z = shrink(_Dz(x) + udel_z, lamb_1 / rho)
            udel_x = _Dx(x) - del_x + udel_x
            udel_y = _Dy(x) - del_y + udel_y
            udel_z = _Dz(x) - del_z + udel_z
        return x

    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=outer_iter)
                return x, x_mean
        return ADMM_TV_fn

    mc_update_fn = get_ADMM_TV_fn()

    def ADMM_TV(data, measurement=None):
        with torch.no_grad():
            x = torch.zeros(data.shape).to(data.device)
            x = mc_update_fn(x, measurement=measurement)
            return x

    return ADMM_TV


def get_ADMM_TV_isotropic(eps=1e-5, radon=None, save_progress=False, save_root=None,
                          img_shape=None, lamb_1=5, rho=10, outer_iter=30, inner_iter=20):
    """
    (get_ADMM_TV): implements anisotropic TV-ADMM
    In contrast, this function implements isotropic TV, which regularizes with |TV|_{1,2}
    """
    del_x = torch.zeros(img_shape)
    del_y = torch.zeros(img_shape)
    del_z = torch.zeros(img_shape)
    udel_x = torch.zeros(img_shape)
    udel_y = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def A_cg(x):
        return _AT(_A(x)) + rho * (_DxT(_Dx(x)) + _DyT(_Dy(x)) + _DzT(_Dz(x)))

    
    def CG(A_fn, b_cg, x, n_inner=20):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(x, ATy, niter=30):
        nonlocal del_x, del_y, del_z, udel_x, udel_y, udel_z
        if del_z.device != x.device:
            del_x = del_x.to(x.device)
            del_y = del_y.to(x.device)
            del_z = del_z.to(x.device)
            udel_x = udel_x.to(x.device)
            udel_y = udel_y.to(x.device)
            udel_z = udel_z.to(x.device)
        for i in tqdm(range(niter)):
            b_cg = ATy + rho * ((_DxT(del_x) - _DxT(udel_x))
                                + (_DyT(del_y) - _DyT(udel_y))
                                + (_DzT(del_z) - _DzT(udel_z)))
            x = CG(A_cg, b_cg, x, n_inner=inner_iter)
            if save_progress:
                plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear(x[0:1]), cmap='gray')

            # Each of shape [448, 1, 256, 256]
            _Dxx = _Dx(x)
            _Dyx = _Dy(x)
            _Dzx = _Dz(x)
            # shape [448, 3, 256, 256]. dim=1 gradient dimension
            _Dxa = torch.cat((_Dxx, _Dyx, _Dzx), dim=1)
            udel_a = torch.cat((udel_x, udel_y, udel_z), dim=1)

            # prox
            del_a = prox_l21(_Dxa + udel_a, lamb_1 / rho, dim=1)

            # split
            del_x, del_y, del_z = torch.split(del_a, 1, dim=1)

            # del_x = prox_l21(_Dxx + udel_x, lamb_1 / rho, -2)
            # del_y = prox_l21(_Dyx + udel_y, lamb_1 / rho, -1)
            # del_z = prox_l21(_Dzx + udel_z, lamb_1 / rho, 0)

            udel_x = _Dxx - del_x + udel_x
            udel_y = _Dyx - del_y + udel_y
            udel_z = _Dzx - del_z + udel_z
        return x

    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x = CS_routine(x, ATy, niter=outer_iter)
                return x
        return ADMM_TV_fn

    mc_update_fn = get_ADMM_TV_fn()

    def ADMM_TV(data, measurement=None):
        with torch.no_grad():
            x = torch.zeros(data.shape).to(data.device)
            x = mc_update_fn(x, measurement=measurement)
            return x

    return ADMM_TV

def prox_l21(src, lamb, dim):
    """
    src.shape = [448(z), 1, 256(x), 256(y)]
    """
    weight_src = torch.linalg.norm(src, dim=dim, keepdim=True)
    weight_src_shrink = shrink(weight_src, lamb)

    weight = weight_src_shrink / weight_src
    return src * weight


def shrink(weight_src, lamb):
    return torch.sign(weight_src) * torch.max(torch.abs(weight_src) - lamb, torch.zeros_like(weight_src))


def get_pc_radon_ADMM_TV_mri(sde, predictor, corrector, inverse_scaler, snr, mask=None,
                             n_steps=1, probability_flow=False, continuous=False,
                             denoise=True, eps=1e-5, save_progress=False, save_root=None,
                             img_shape=None, lamb_1=5, rho=10):
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    del_z = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    def _A(x):
        return fft2(x) * mask

    def _AT(kspace):
        return torch.real(ifft2(kspace))

    def _Dz(x):  # Batch direction
        y = torch.zeros_like(x)
        y[:-1] = x[1:]
        y[-1] = x[0]
        return y - x

    def _DzT(x):  # Batch direction
        y = torch.zeros_like(x)
        y[:-1] = x[1:]
        y[-1] = x[0]

        tempt = -(y - x)
        difft = tempt[:-1]
        y[1:] = difft
        y[0] = x[-1] - x[0]

        return y

    def A_cg(x):
        return _AT(_A(x)) + rho * _DzT(_Dz(x))

    def shrink(src, lamb):
        return torch.sign(src) * torch.max(torch.abs(src) - lamb, torch.zeros_like(src))

    def CG(A_fn, b_cg, x, n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(x, ATy, niter=20):
        nonlocal del_z, udel_z
        if del_z.device != x.device:
            del_z = del_z.to(x.device)
            udel_z = del_z.to(x.device)
        for i in range(niter):
            b_cg = ATy + rho * (_DzT(del_z) - _DzT(udel_z))
            x = CG(A_cg, b_cg, x, n_inner=1)

            del_z = shrink(_Dz(x) + udel_z, lamb_1 / rho)
            udel_z = _Dz(x) - del_z + udel_z
        x_mean = x
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=1)
                return x, x_mean
        return ADMM_TV_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    mc_update_fn = get_ADMM_TV_fn()

    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                # 1. batchify into sizes that fit into the GPU
                x_batch = batchfy(x, 20)
                # 2. Run PC step for each batch
                x_agg = list()
                for idx, x_batch_sing in enumerate(x_batch):
                    x_batch_sing, _ = predictor_denoise_update_fn(model, data, x_batch_sing, t)
                    x_batch_sing, _ = corrector_denoise_update_fn(model, data, x_batch_sing, t)
                    x_agg.append(x_batch_sing)
                # 3. Aggregate to run ADMM TV
                x = torch.cat(x_agg, dim=0)
                # 4. Run ADMM TV
                x, x_mean = mc_update_fn(x, measurement=measurement)

                if save_progress:
                    if (i % 50) == 0:
                        print(f'iter: {i}/{sde.N}')
                        plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear(x_mean[0:1]), cmap='gray')

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon