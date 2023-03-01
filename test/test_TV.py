"""
python -m pytest
"""
import sys

import pytest
import torch
import matplotlib.pyplot as plt
import skimage


import controllable_generation_TV as TV

@pytest.mark.parametrize(
    ["A", "AT"],
    [
        [TV._Dz, TV._DzT],
        [TV._Dx, TV._DxT],
        [TV._Dy, TV._DyT],
    ]
)
def test_adjoint(A, AT):
    x = torch.randn(10, 10, 10, 10)
    y = torch.randn(10, 10, 10, 10)

    torch.testing.assert_allclose(
        torch.dot(A(x).ravel(), y.ravel()),
        torch.dot(x.ravel(), AT(y).ravel())
    )

def test_prox_l21():
    prox_val = .75

    Dx = torch.randn(1, 1, 1, 1)
    Dy = torch.randn(1, 1, 1, 1)
    Dz = torch.randn(1, 1, 1, 1)

    Dq = torch.cat((Dx, Dy, Dz), dim=1)
    Dq_norm = torch.linalg.norm(Dq)

    Dq_prox = TV.prox_l21(Dq, prox_val, dim=1)
    Dq_prox_norm = torch.linalg.norm(Dq_prox)

    torch.testing.assert_allclose(
        max(Dq_norm, 0) - prox_val,
        Dq_prox_norm,
    )
    torch.testing.assert_allclose(
        Dq / Dq_norm,
        Dq_prox / Dq_prox_norm,
    )


class Identity:
    @staticmethod
    def A(x):
        return x

    @staticmethod    
    def AT(y):
        return y

def test_ADMM_TV_isotropic():
    x_gt = skimage.data.astronaut().mean(axis=2) / 255

    x_gt = torch.tensor(x_gt).reshape((1, 1) + x_gt.shape)
    y = x_gt + 0.5 * torch.randn_like(x_gt)

    x0 = torch.zeros_like(y)

    ADMM_TV = TV.get_ADMM_TV_isotropic(
        radon=Identity(), img_shape=y.shape,
        lamb_1 = 1e0, rho=1e2)

    x_recon = ADMM_TV(x0, y)

    args = dict(vmin=-0.2, vmax=1.2)
    
    fig, ax = plt.subplots()
    im = ax.imshow(x_gt.squeeze(), **args)
    fig.colorbar(im)
    fig.savefig('x_gt.png')

    fig, ax = plt.subplots()
    im = ax.imshow(y.squeeze(), **args)
    fig.colorbar(im)
    fig.savefig('y.png')

    fig, ax = plt.subplots()
    im = ax.imshow(x_recon.squeeze(), **args)
    fig.colorbar(im)
    fig.savefig('x_recon.png')    