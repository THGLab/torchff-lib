"""Parity tests for multipole local-frame rotation: Python reference vs CUDA custom op."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("openmm")

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from torchff.multipoles import MultipolarRotation

ROOT = Path(__file__).resolve().parent


def amoeba_water_multipole_frame_data(
    pdb_path: Path,
    *,
    cutoff_nm: float,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load a water box PDB and return only data needed for local rotation matrices.

    Parameters
    ----------
    pdb_path
        Path to an OpenMM-readable PDB (e.g. AMOEBA water box).
    cutoff_nm
        Nonbonded cutoff when building the system; must be below half the shortest box
        edge for small cells.
    dtype
        Floating dtype for coordinates.

    Returns
    -------
    coords_nm : torch.Tensor
        Shape ``(N, 3)``, positions in nanometers.
    z_atoms, x_atoms, y_atoms : torch.Tensor
        Shape ``(N,)``, ``torch.long``, multipole local-frame atom indices.
    axis_types : torch.Tensor
        Shape ``(N,)``, ``torch.long``, OpenMM axis type integers.
    """
    pdb = app.PDBFile(str(pdb_path))
    topology = pdb.topology
    ff = app.ForceField("amoeba2018.xml")
    system = ff.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=None,
        rigidWater=False,
        polarization="direct",
    )
    mp_force = next(f for f in system.getForces() if isinstance(f, mm.AmoebaMultipoleForce))
    z_list: list[int] = []
    x_list: list[int] = []
    y_list: list[int] = []
    ax_list: list[int] = []
    for idx in range(mp_force.getNumMultipoles()):
        (
            _charge,
            _molecular_dipole,
            _molecular_quadrupole,
            axis_type,
            multipole_atom_z,
            multipole_atom_x,
            multipole_atom_y,
            _thole,
            _damping_factor,
            _polarity,
        ) = mp_force.getMultipoleParameters(idx)
        ax_list.append(int(axis_type))
        z_list.append(int(multipole_atom_z))
        x_list.append(int(multipole_atom_x))
        y_list.append(int(multipole_atom_y))

    coords_nm = torch.tensor(
        pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=dtype,
    )
    z_atoms = torch.tensor(z_list, dtype=torch.long)
    x_atoms = torch.tensor(x_list, dtype=torch.long)
    y_atoms = torch.tensor(y_list, dtype=torch.long)
    axis_types = torch.tensor(ax_list, dtype=torch.long)
    return coords_nm, z_atoms, x_atoms, y_atoms, axis_types


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for MultipolarRotation with use_customized_ops=True",
)
def test_rotation_matrices_match_reference_water():
    """AMOEBA water: CUDA custom op vs batched Python reference."""
    pdb = ROOT / "water" / "water_100.pdb"
    coords_nm, z, x, y, ax = amoeba_water_multipole_frame_data(pdb, cutoff_nm=0.15)
    dtype = torch.float64
    device = torch.device("cuda")
    coords = coords_nm.to(device=device, dtype=dtype)
    z = z.to(device=device)
    x = x.to(device=device)
    y = y.to(device=device)
    ax = ax.to(device=device)

    rot_ref = MultipolarRotation.compute_matrices(
        coords, z, x, y, ax, use_customized_ops=False
    )
    rot_cuda = MultipolarRotation.compute_matrices(
        coords, z, x, y, ax, use_customized_ops=True
    )

    assert torch.allclose(rot_ref, rot_cuda, rtol=1e-5, atol=1e-6)
