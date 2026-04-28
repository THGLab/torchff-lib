import os
import json

# Disable torch.compile/dynamo via environment for this script.
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch

import openmm as mm
import openmm.app as app
import openmm.unit as unit

import sys
test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, '../../'))
from torchff.multipolar.rotation import (
    _compute_rotation_matrices_python,
    rotateDipoles,
    rotateQuadrupoles,
)

device = torch.device('cpu')
dtype = torch.float64

# Resolve PDB path relative to this test file.

pdb_path = os.path.join(test_dir, "water_2.pdb")

pdb = app.PDBFile(pdb_path)

forcefield = app.ForceField("amoeba2018.xml")

system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.NoCutoff,
    nonbondedCutoff=1.0*unit.nanometer,
    constraints=None,
    rigidWater=False,
)

multipole_force = None
for force in system.getForces():
    if isinstance(force, mm.AmoebaMultipoleForce):
        multipole_force = force
        break

num_multipoles = multipole_force.getNumMultipoles()
charges = []
dipoles_local = []
quads_local = []
axis_types = []
z_atoms = []
x_atoms = []
y_atoms = []

for idx in range(num_multipoles):
    (
        charge,
        molecular_dipole,
        molecular_quadrupole,
        axis_type,
        multipole_atom_z,
        multipole_atom_x,
        multipole_atom_y,
        thole,
        damping_factor,
        polarity,
    ) = multipole_force.getMultipoleParameters(idx)

    # Convert OpenMM quantities to plain floats in consistent units.
    if isinstance(charge, unit.Quantity):
        charge_val = charge.value_in_unit(unit.elementary_charge)
    else:
        charge_val = float(charge)
    charges.append(charge_val)

    # OpenMM returns local-frame dipoles (charge * distance).
    dip_vec = []
    for comp in molecular_dipole:
        if isinstance(comp, unit.Quantity):
            dip_vec.append(comp.value_in_unit(unit.elementary_charge * unit.nanometer))
        else:
            dip_vec.append(float(comp))
    dipoles_local.append(dip_vec)

    # molecular_quadrupole is provided as a flat sequence of 9 components
    # in the local frame (charge * distance^2). Convert to 3x3 matrix.
    quad_flat = []
    for comp in molecular_quadrupole:
        if isinstance(comp, unit.Quantity):
            quad_flat.append(
                comp.value_in_unit(unit.elementary_charge * unit.nanometer * unit.nanometer) * 3
            )
        else:
            quad_flat.append(float(comp) * 3)
    qmat = [
        quad_flat[0:3],
        quad_flat[3:6],
        quad_flat[6:9],
    ]
    quads_local.append(qmat)

    axis_types.append(int(axis_type))
    z_atoms.append(int(multipole_atom_z))
    x_atoms.append(int(multipole_atom_x))
    y_atoms.append(int(multipole_atom_y))

# Coordinates and box (in nm) as PyTorch tensors.
pos_nm = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
coords = torch.tensor(pos_nm, dtype=dtype, device=device)

q = torch.tensor(charges, dtype=dtype, device=device)
p_local = torch.tensor(dipoles_local, dtype=dtype, device=device)
t_local = torch.tensor(quads_local, dtype=dtype, device=device)


axis_types_t = torch.tensor(axis_types, dtype=torch.int32, device=device)
z_atoms_t = torch.tensor(z_atoms, dtype=torch.int64, device=device)
x_atoms_t = torch.tensor(x_atoms, dtype=torch.int64, device=device)
y_atoms_t = torch.tensor(y_atoms, dtype=torch.int64, device=device)

# Compute local->global rotation matrices and rotate dipoles/quadrupoles.
rot_mats = _compute_rotation_matrices_python(
    coords,
    z_atoms_t,
    x_atoms_t,
    y_atoms_t,
    axis_types_t,
)
p_global = rotateDipoles(p_local, rot_mats).squeeze(1)
t_global = rotateQuadrupoles(t_local, rot_mats)

# Build only intermolecular pairs: Cartesian product of atoms 0–2 (water 1)
# with atoms 3–5 (water 2). This excludes all intramolecular pairs.
if coords.shape[0] != 6:
    raise RuntimeError("This test currently assumes a water dimer with 6 atoms.")
idx_i = torch.arange(0, 3, device=device, dtype=torch.int64)
idx_j = torch.arange(3, 6, device=device, dtype=torch.int64)
ii, jj = torch.meshgrid(idx_i, idx_j, indexing="ij")
pairs = torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=1)

# TorchFF uses atomic units internally: distances in Bohr, energy in Hartree.
nm_to_bohr = (1.0 * unit.nanometer).value_in_unit(unit.bohr)
coords_bohr = coords * nm_to_bohr
p_bohr = p_global * nm_to_bohr
t_bohr2 = t_global * (nm_to_bohr**2)

output_path = os.path.join(test_dir, "water_amoeba_multipoles.json")

# Prepare JSON-serializable data.  We store:
# - "coordinates": original positions in nm
# - "coords": positions in Bohr
# - "pairs": intermolecular atom pairs
# - "charges": monopoles in units of e
# - "dipoles": global-frame dipoles in atomic units (e * Bohr)
# - "quadrupoles": global-frame quadrupoles in atomic units (e * Bohr^2)
data = {
    "coords": coords_bohr.tolist(),
    "pairs": pairs.tolist(),
    "charges": charges,
    "dipoles": p_bohr.tolist(),
    "quadrupoles": t_bohr2.tolist(),
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Wrote AMOEBA water dimer multipole data to {output_path}")

