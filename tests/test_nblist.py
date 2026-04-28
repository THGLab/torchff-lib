import os
import pytest
from tqdm import tqdm
import time
import numpy as np
import torch
import torch.nn as nn
import torchff
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from torchff.test_utils import perf_op
from torchff.pbc import PBC
from torchff.nblist import NeighborList


def create_benchmark_data(num: int, density: float, cutoff, device, dtype):
    boxLen = (num / density) ** (1/3) # density is number of particles / volume
    coords = np.random.rand(num, 3) * boxLen
    coords_tensor = torch.from_numpy(coords).to(device=device, dtype=dtype)
    box_tensor = torch.tensor([[boxLen, 0, 0], [0, boxLen, 0], [0, 0, boxLen]], device=device, dtype=dtype)
    max_npairs = int(num * ((cutoff**3) * density) / 2 * 1.2)
    return coords_tensor, box_tensor, cutoff, max_npairs


def _pairs_to_canonical_set(pairs: torch.Tensor):
    """Normalize pairs to (min, max) tuples and return as a Python set."""
    if pairs.shape[0] == 0:
        return set()
    lo = torch.min(pairs, dim=1).values
    hi = torch.max(pairs, dim=1).values
    return set(zip(lo.cpu().tolist(), hi.cpu().tolist()))


class TestNeighborListForward:
    """Test that NeighborList.forward executes successfully."""

    @pytest.fixture(params=[
        (50, 100.0, 0.5, 'cpu', torch.float64),
        (100, 100.0, 0.8, 'cpu', torch.float32),
        (1000, 100.0, 0.5, 'cuda', torch.float64),
    ])
    def nblist_data(self, request):
        num, density, cutoff, device, dtype = request.param
        coords, box, cutoff, _ = create_benchmark_data(num, density, cutoff, device, dtype)
        use_custom = (device == 'cuda')
        return coords, box, cutoff, num, device, use_custom

    def test_forward_returns_tensor(self, nblist_data):
        coords, box, cutoff, natoms, device, use_custom = nblist_data
        nblist = NeighborList(natoms, use_customized_ops=use_custom).to(device)
        result = nblist(coords, box, cutoff)
        assert isinstance(result, torch.Tensor)

    def test_forward_output_shape(self, nblist_data):
        coords, box, cutoff, natoms, device, use_custom = nblist_data
        nblist = NeighborList(natoms, use_customized_ops=use_custom).to(device)
        result = nblist(coords, box, cutoff)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_forward_pairs_within_range(self, nblist_data):
        coords, box, cutoff, natoms, device, use_custom = nblist_data
        nblist = NeighborList(natoms, use_customized_ops=use_custom).to(device)
        result = nblist(coords, box, cutoff)
        assert result.shape[0] <= natoms * (natoms - 1) // 2

    def test_forward_no_self_pairs(self, nblist_data):
        coords, box, cutoff, natoms, device, use_custom = nblist_data
        nblist = NeighborList(natoms, include_self=False, use_customized_ops=use_custom).to(device)
        result = nblist(coords, box, cutoff)
        if result.shape[0] > 0:
            assert (result[:, 0] != result[:, 1]).all()

    def test_forward_large_cutoff_finds_all_pairs(self):
        """A cutoff larger than the box should return all pairs."""
        natoms = 20
        coords, box, cutoff, _ = create_benchmark_data(natoms, 100.0, 0.5, 'cpu', torch.float64)
        large_cutoff = box[0, 0].item() * 10
        nblist = NeighborList(natoms, use_customized_ops=False)
        result = nblist(coords, box, large_cutoff)
        assert result.shape[0] == natoms * (natoms - 1) // 2

    def test_forward_zero_cutoff_finds_no_pairs(self):
        natoms = 30
        coords, box, _, _ = create_benchmark_data(natoms, 100.0, 0.5, 'cpu', torch.float64)
        nblist = NeighborList(natoms, use_customized_ops=False)
        result = nblist(coords, box, cutoff=0.0)
        assert result.shape[0] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestNeighborListCustomizedOps:
    """Compare NeighborList results with and without customized CUDA ops."""

    @pytest.fixture(params=[
        (50, 100.0, 0.5, torch.float64, False),
        (200, 100.0, 0.8, torch.float64, False),
        (500, 100.0, 0.5, torch.float32, False),
        (50, 100.0, 0.5, torch.float64, True),
        (200, 100.0, 0.8, torch.float64, True),
    ])
    def nblist_pair_data(self, request):
        num, density, cutoff, dtype, include_self = request.param
        coords, box, cutoff, _ = create_benchmark_data(num, density, cutoff, 'cuda', dtype)
        return coords, box, cutoff, num, include_self

    def test_customized_ops_match_python(self, nblist_pair_data):
        """Pairs from CUDA ops must match the pure-Python reference."""
        coords, box, cutoff, natoms, include_self = nblist_pair_data

        nblist_py = NeighborList(natoms, include_self=include_self, use_customized_ops=False).to('cuda')
        nblist_cpp = NeighborList(natoms, include_self=include_self, use_customized_ops=True).to('cuda')

        pairs_py = nblist_py(coords, box, cutoff)
        pairs_cpp = nblist_cpp(coords, box, cutoff)

        set_py = _pairs_to_canonical_set(pairs_py)
        set_cpp = _pairs_to_canonical_set(pairs_cpp)

        assert set_py == set_cpp, (
            f"Pair count mismatch: python={len(set_py)}, cpp={len(set_cpp)}; "
            f"only_python={len(set_py - set_cpp)}, only_cpp={len(set_cpp - set_py)}"
        )
    
    def test_customized_ops_match_python_non_periodic(self, nblist_pair_data):
        """Pairs from CUDA ops must match the pure-Python reference."""
        coords, box, cutoff, natoms, include_self = nblist_pair_data

        nblist_py = NeighborList(natoms, include_self=include_self, use_customized_ops=False).to('cuda')
        nblist_cpp = NeighborList(natoms, include_self=include_self, use_customized_ops=True).to('cuda')

        pairs_py = nblist_py(coords, None, cutoff)
        pairs_cpp = nblist_cpp(coords, None, cutoff)

        set_py = _pairs_to_canonical_set(pairs_py)
        set_cpp = _pairs_to_canonical_set(pairs_cpp)

        assert set_py == set_cpp, (
            f"Pair count mismatch: python={len(set_py)}, cpp={len(set_cpp)}; "
            f"only_python={len(set_py - set_cpp)}, only_cpp={len(set_cpp - set_py)}"
        )

    def test_customized_ops_pair_count(self, nblist_pair_data):
        """Both paths should find the same number of pairs."""
        coords, box, cutoff, natoms, include_self = nblist_pair_data

        nblist_py = NeighborList(natoms, include_self=include_self, use_customized_ops=False).to('cuda')
        nblist_cpp = NeighborList(natoms, include_self=include_self, use_customized_ops=True).to('cuda')

        pairs_py = nblist_py(coords, box, cutoff)
        pairs_cpp = nblist_cpp(coords, box, cutoff)

        assert pairs_py.shape[0] == pairs_cpp.shape[0]

    def test_customized_ops_large_cutoff(self):
        """With a very large cutoff both paths should return all pairs."""
        natoms = 30
        coords, box, _, _ = create_benchmark_data(natoms, 100.0, 0.5, 'cuda', torch.float64)
        large_cutoff = box[0, 0].item() * 10

        nblist_py = NeighborList(natoms, use_customized_ops=False).to('cuda')
        nblist_cpp = NeighborList(natoms, use_customized_ops=True).to('cuda')

        pairs_py = nblist_py(coords, box, large_cutoff)
        pairs_cpp = nblist_cpp(coords, box, large_cutoff)

        expected = natoms * (natoms - 1) // 2
        assert pairs_py.shape[0] == expected
        assert pairs_cpp.shape[0] == expected

    def test_customized_ops_zero_cutoff(self):
        """Zero cutoff should yield no pairs on both paths."""
        natoms = 30
        coords, box, _, _ = create_benchmark_data(natoms, 100.0, 0.5, 'cuda', torch.float64)

        nblist_py = NeighborList(natoms, use_customized_ops=False).to('cuda')
        nblist_cpp = NeighborList(natoms, use_customized_ops=True).to('cuda')

        pairs_py = nblist_py(coords, box, cutoff=0.0)
        pairs_cpp = nblist_cpp(coords, box, cutoff=0.0)

        assert pairs_py.shape[0] == 0
        assert pairs_cpp.shape[0] == 0



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCellListAlgorithm:
    """Verify that the cell-list algorithm produces identical results to nsquared."""

    @pytest.fixture(params=[
        # (1000, 100.0, 0.5, torch.float64, False),
        # (2000, 100.0, 0.8, torch.float64, False),
        # (5000, 100.0, 0.5, torch.float32, False),
        (1000000, 100.0, 0.8, torch.float64, False),
        # (1000, 100.0, 0.5, torch.float64, True),
        # (5000, 100.0, 0.5, torch.float64, True),
    ])
    def cell_list_data(self, request):
        num, density, cutoff, dtype, include_self = request.param
        coords, box, cutoff, _ = create_benchmark_data(num, density, cutoff, 'cuda', dtype)
        return coords, box, cutoff, num, include_self

    # def test_cell_list_matches_nsquared(self, cell_list_data):
    #     """Pair sets from cell_list and nsquared must be identical."""
    #     coords, box, cutoff, natoms, include_self = cell_list_data

    #     nblist_nsq = NeighborList(
    #         natoms, include_self=include_self,
    #         use_customized_ops=True, algorithm='nsquared'
    #     ).to('cuda')
    #     nblist_cl = NeighborList(
    #         natoms, include_self=include_self,
    #         use_customized_ops=True, algorithm='cell_list'
    #     ).to('cuda')

    #     pairs_nsq = nblist_nsq(coords, box, cutoff)
    #     pairs_cl = nblist_cl(coords, box, cutoff)

    #     set_nsq = _pairs_to_canonical_set(pairs_nsq)
    #     set_cl = _pairs_to_canonical_set(pairs_cl)

    #     assert set_nsq == set_cl, (
    #         f"Pair count mismatch: nsquared={len(set_nsq)}, cell_list={len(set_cl)}; "
    #         f"only_nsquared={len(set_nsq - set_cl)}, only_cell_list={len(set_cl - set_nsq)}"
    #     )

    def test_cell_list_pair_count(self, cell_list_data):
        """Both algorithms must find the same number of pairs."""
        coords, box, cutoff, natoms, include_self = cell_list_data

        nblist_nsq = NeighborList(
            natoms, include_self=include_self,
            use_customized_ops=True, algorithm='nsquared'
        ).to('cuda')
        nblist_cl = NeighborList(
            natoms, include_self=include_self,
            use_customized_ops=True, algorithm='cell_list'
        ).to('cuda')

        max_pairs = natoms * 1000

        pairs_nsq = nblist_nsq(coords, box, cutoff, max_pairs)
        pairs_cl = nblist_cl(coords, box, cutoff, max_pairs)

        assert pairs_nsq.shape[0] == pairs_cl.shape[0], (
            f"Count mismatch: nsquared={pairs_nsq.shape[0]}, cell_list={pairs_cl.shape[0]}"
        )

    # def test_cell_list_matches_nsquared_with_exclusions(self):
    #     """Cell-list with exclusions must match nsquared with same exclusions."""
    #     natoms = 1000
    #     coords, box, cutoff, _ = create_benchmark_data(natoms, 100.0, 0.5, 'cuda', torch.float64)

    #     n_excl = natoms // 3
    #     excl_pairs = torch.stack([
    #         torch.arange(0, n_excl * 3, 3, device='cuda'),
    #         torch.arange(1, n_excl * 3, 3, device='cuda'),
    #     ], dim=1)

    #     nblist_nsq = NeighborList(
    #         natoms, exclusions=excl_pairs,
    #         use_customized_ops=True, algorithm='nsquared'
    #     ).to('cuda')
    #     nblist_cl = NeighborList(
    #         natoms, exclusions=excl_pairs,
    #         use_customized_ops=True, algorithm='cell_list'
    #     ).to('cuda')

    #     pairs_nsq = nblist_nsq(coords, box, cutoff)
    #     pairs_cl = nblist_cl(coords, box, cutoff)

    #     set_nsq = _pairs_to_canonical_set(pairs_nsq)
    #     set_cl = _pairs_to_canonical_set(pairs_cl)

    #     assert set_nsq == set_cl, (
    #         f"With exclusions - nsquared={len(set_nsq)}, cell_list={len(set_cl)}; "
    #         f"only_nsquared={len(set_nsq - set_cl)}, only_cell_list={len(set_cl - set_nsq)}"
    #     )


# def test_benchmark_nblist_nsq(device='cuda', dtype=torch.float64):
#     density = 100 # liq water is 100 atoms/nm^3
#     cutoff = 0.8
#     for num in [1e3, 3e3, 1e4, 3e4, 1e5, 3e5]:
#         coords, box, cutoff, max_npairs = create_benchmark_data(int(num), density, cutoff, device, dtype)
#         try:
#             perfs = perf_op(torchff.build_neighbor_list_nsquared, coords, box, cutoff, max_npairs, False)
#             print(f"Number of atoms: {int(num)}, cutoff: {cutoff}, box: {box[0,0].item()}, Time: {np.mean(perfs)}+-{np.std(perfs)} ms")
#         except Exception as e:
#             print(f"Number of atoms: {int(num)}, cutoff: {cutoff}, box: {box[0,0].item()}, Failed because {e}")



# def test_nblist_cell_list():
#     dirname = os.path.dirname(__file__)
#     pdb = app.PDBFile(os.path.join(dirname, 'water/water_10000.pdb'))
#     top = pdb.getTopology()
#     boxVectors = top.getPeriodicBoxVectors()
#     print(boxVectors)
#     box = torch.tensor([
#         [boxVectors[0].x, boxVectors[0].y, boxVectors[0].z],
#         [boxVectors[1].x, boxVectors[1].y, boxVectors[1].z],
#         [boxVectors[2].x, boxVectors[2].y, boxVectors[2].z]
#     ], dtype=torch.float32, device='cuda', requires_grad=False)
#     coords = torch.tensor(
#         pdb.getPositions(asNumpy=True)._value.tolist(),
#         dtype=torch.float32, device='cuda', requires_grad=True
#     )

#     cutoff = 0.4

#     pairs_nsquared, num = torchff.build_neighbor_list_nsquared(coords, box, cutoff, -1, False)
#     print(pairs_nsquared.shape, num)
#     mask = torch.floor_divide(pairs_nsquared[:, 0], 3) != torch.floor_divide(pairs_nsquared[:, 1], 3)
#     pairs_nsquared = pairs_nsquared[mask]

    

#     # pairs_clist, _ = torchff.build_neighbor_list_cell_list(coords, box, cutoff, coords.shape[0] * 500, 0.5, False, True)
#     excl_i, excl_j = [], []
#     for n in range(top.getNumAtoms()//3):
#         for i in range(3):
#             for j in range(3):
#                 excl_i.append(n*3+i)
#                 excl_j.append(n*3+j)
#     exclusions = torch.tensor([excl_i, excl_j], dtype=torch.int64, device='cuda')

#     pairs_clist, _ = torchff.build_neighbor_list_cluster_pairs(
#         coords, box, cutoff, exclusions, 0.4, -1, -1, False
#     )

#     # pairs_clist_set = set()
#     # for p in pairs_clist.detach().cpu().numpy().tolist():
#     #     if (min(p), max(p)) in pairs_clist_set:
#     #         print(f"Duplicated: {(min(p), max(p))}")
#     #     pairs_clist_set.add((min(p), max(p)))
    
#     pairs_nsquared_set = set((min(p), max(p)) for p in pairs_nsquared.detach().cpu().numpy().tolist())
#     print(len(pairs_nsquared_set))
#     pairs_clist_set = set((min(p), max(p)) for p in pairs_clist.detach().cpu().numpy().tolist())
#     print(len(pairs_clist_set))
#     clist_diff_set = pairs_clist_set.difference(pairs_nsquared_set)
#     if len(clist_diff_set) > 0:
#         print("The fist 10 pairs are in cell-list nblist but not in nsquared:", list(clist_diff_set)[:10])
#         clist_diff = torch.tensor(list(clist_diff_set), dtype=torch.int32)
#         print(clist_diff)
#         diff_vec = coords[clist_diff[:, 0]] - coords[clist_diff[:, 1]]
#         print(torch.norm(((diff_vec / box[0][0]) - torch.round(diff_vec / box[0][0])) * box[0][0], dim=1))
#     nsq_diff_set = pairs_nsquared_set.difference(pairs_clist_set)

#     if len(nsq_diff_set) > 0:
#         print("The first 10 pairs are in nsquared nblist but not in cell-list:", list(nsq_diff_set)[:10])
#         nsq_diff = torch.tensor(list(nsq_diff_set), dtype=torch.int32)
#         diff_vec = coords[nsq_diff[:, 0]] - coords[nsq_diff[:, 1]]
#         print(torch.norm(((diff_vec / box[0][0]) - torch.round(diff_vec / box[0][0])) * box[0][0], dim=1))

#     assert pairs_nsquared.shape[0] == pairs_clist.shape[0]
           
#     # perf_op(
#     #     torchff.build_neighbor_list_nsquared,
#     #     coords, box, cutoff, coords.shape[0]*500,
#     #     desc='O(N^2) NBList',
#     #     run_backward=False,
#     #     use_cuda_graph=False,
#     #     explicit_sync=True
#     # )
#     # perf_op(
#     #     torchff.build_neighbor_list_cluster_pairs,
#     #     coords, box, cutoff, None, 0.5, -1, -1, False,
#     #     desc='Cluster Pair NBList',
#     #     run_backward=False,
#     #     use_cuda_graph=False,
#     #     explicit_sync=True
#     # )
    

# def test_build_cluster_pair_perf():
#     dirname = os.path.dirname(__file__)
#     pdb = app.PDBFile(os.path.join(dirname, 'water/water_10000.pdb'))
#     cutoff = 1.2

#     top = pdb.getTopology()
#     boxVectors = top.getPeriodicBoxVectors()
#     print(boxVectors)
#     box = torch.tensor([
#         [boxVectors[0].x, boxVectors[0].y, boxVectors[0].z],
#         [boxVectors[1].x, boxVectors[1].y, boxVectors[1].z],
#         [boxVectors[2].x, boxVectors[2].y, boxVectors[2].z]
#     ], dtype=torch.float32, device='cuda', requires_grad=False)
#     coords = torch.tensor(
#         pdb.getPositions(asNumpy=True)._value.tolist(),
#         dtype=torch.float32, device='cuda', requires_grad=True
#     )
#     ff = app.ForceField('tip3p.xml')
#     system: mm.System = ff.createSystem(
#         top,
#         nonbondedMethod=app.PME,
#         nonbondedCutoff=cutoff*unit.nanometer,
#         constraints=None,
#         rigidWater=False
#     )

   
#     # water excls
#     excl_i, excl_j = [], []
#     for n in range(system.getNumParticles()//3):
#         for i in range(3):
#             for j in range(3):
#                 excl_i.append(n*3+i)
#                 excl_j.append(n*3+j)
#     exclusions = torch.tensor([excl_i, excl_j], dtype=torch.int64, device='cuda')

#     perf_op(
#         torchff.build_cluster_pairs,
#         coords, box, cutoff, exclusions, 0.6, -1,
#         use_cuda_graph=False,
#         run_backward=False
#     )