#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>


TORCH_LIBRARY_FRAGMENT(torchff, m) {
  m.def(
    "cmm_non_elec_nonbonded_interaction_from_pairs("
      "Tensor coords, Tensor box, Tensor pairs, Tensor multipoles, "
      "Tensor q_pauli, Tensor Kdipo_pauli, Tensor Kquad_pauli, Tensor b_pauli_ij, "
      "Tensor q_xpol, Tensor Kdipo_xpol, Tensor Kquad_xpol, Tensor b_xpol_ij, "
      "Tensor q_ct_don, Tensor Kdipo_ct_don, Tensor Kquad_ct_don, "
      "Tensor q_ct_acc, Tensor Kdipo_ct_acc, Tensor Kquad_ct_acc, Tensor b_ct_ij, Tensor eps_ct_ij, "
      "Tensor C6_disp_ij, Tensor b_disp_ij, "
      "Scalar rcut_sr, Scalar rcut_lr, Scalar rcut_switch_buf"
    ") -> (Tensor, Tensor)"
  );
  m.def(
    "cmm_elec_from_pairs("
      "Tensor coords, Tensor box, "
      "Tensor pairs, Tensor pairs_excl, "
      "Tensor multipoles, "
      "Tensor Z, Tensor b_elec_ij, Tensor b_elec, "
      "Scalar ewald_alpha, "
      "Scalar rcut_sr, Scalar rcut_lr, Scalar rcut_switch_buf"
    ") -> (Tensor, Tensor, Tensor)"
  );
  m.def(
    "cmm_field_dependent_morse_bond("
      "Tensor coords, Tensor bonds, "
      "Tensor req_0, Tensor kb_0, Tensor D, "
      "Tensor dipole_deriv_1, Tensor dipole_deriv_2, "
      "Tensor efield"
    ") -> Tensor"
  );
  m.def(
    "cmm_bond_charge_flux("
      "Tensor coords, Tensor bonds, Tensor req, "
      "Tensor j_cf, Tensor j_cf_pauli "
    ") -> (Tensor, Tensor)"
  );
  m.def(
    "cmm_angles("
      "Tensor coords, Tensor angles, "
      "Tensor theta_0, Tensor k_theta, Tensor r_eq_1, Tensor r_eq_2, "
      "Tensor k_bb, Tensor k_ba_1, Tensor k_ba_2, "
      "Tensor j_cf_bb, Tensor j_cf_angle, Scalar ene_coupling_min"
    ") -> (Tensor, Tensor)"
  );
  m.def(
    "compute_cmm_polarization_real_space("
      "Tensor coords, Tensor box, Tensor pairs, Tensor pairs_excl, Tensor b_elec_ij, Tensor vec_in, "
      "Scalar ewald_alpha, Scalar rcut_sr, Scalar rcut_lr, "
      "Tensor(a!) vec_out"
    ") -> ()"
  );
  m.def(
    "cmm_polarization_energy_from_induced_multipoles("
      "Tensor coords, Tensor box, Tensor pairs, Tensor pairs_excl, "
      "Tensor induced_multipoles, Tensor b_elec_ij, "
      "Scalar ewald_alpha, Scalar rcut_sr, Scalar rcut_lr, Scalar natoms "
    ") -> Tensor"
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchff CMM CUDA extension";
}
