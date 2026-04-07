import pytest
import random
import torch
from torchff.angle import *

from torchff.test_utils import check_op


@pytest.mark.dependency()
@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64)
])
def test_harmonic_angle(device, dtype):
    requires_grad = True
    N = 100
    Nangles = N
    arange = list(range(N))
    angles = torch.tensor([random.sample(arange, 3) for _ in range(Nangles)], device=device)
    coords = torch.rand(N*3, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    theta0 = torch.rand(Nangles, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.rand(Nangles, device=device, dtype=dtype, requires_grad=requires_grad)

    func = HarmonicAngle(use_customized_ops=True)
    func_ref = HarmonicAngle(use_customized_ops=False)

    check_op(
        func,
        func_ref,
        {'coords': coords, 'angles': angles, 'theta0': theta0, 'k': k},
        check_grad=True,
        atol=1e-5,
    )

    forces = torch.zeros_like(coords, requires_grad=False)
    compute_harmonic_angle_forces(coords, angles, theta0, k, forces)
    coords.grad = None
    e = func_ref(coords, angles, theta0, k)
    e.backward()
    assert torch.allclose(forces, -coords.grad.clone().detach(), atol=1e-5)


@pytest.mark.dependency()
@pytest.mark.parametrize("device, dtype", [
    ('cuda', torch.float32),
    ('cuda', torch.float64)
])
def test_amoeba_angle(device, dtype):
    requires_grad = True
    N = 100
    Nangles = N
    arange = list(range(N))
    angles = torch.tensor([random.sample(arange, 3) for _ in range(Nangles)], device=device)
    coords = torch.rand(N * 3, 3, requires_grad=requires_grad, device=device, dtype=dtype)
    theta0 = torch.rand(Nangles, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.rand(Nangles, device=device, dtype=dtype, requires_grad=requires_grad)

    func = AmoebaAngle(use_customized_ops=True)
    func_ref = AmoebaAngle(use_customized_ops=False)

    check_op(
        func,
        func_ref,
        {'coords': coords, 'angles': angles, 'theta0': theta0, 'k': k},
        check_grad=True,
        atol=1e-5,
    )