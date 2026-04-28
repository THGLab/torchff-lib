import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchff_nblist
from .pbc import PBC


class NeighborList(nn.Module):
    def __init__(
        self,
        natoms: int,
        exclusions: torch.Tensor | None = None,
        include_self: bool = False,
        use_customized_ops: bool = True,
        algorithm: str = "nsquared",
    ):
        super().__init__()
        if algorithm not in ("nsquared", "cell_list"):
            raise ValueError(
                f"algorithm must be 'nsquared' or 'cell_list', got '{algorithm}'"
            )
        self.natoms = natoms
        self.include_self = include_self
        self.use_customized_ops = use_customized_ops
        self.algorithm = algorithm

        if exclusions is not None:
            self.register_buffer("exclusions_coo", exclusions)
            row_ptr, col_indices = self.convert_pairs_coo_to_csr(exclusions, natoms)
            self.register_buffer("excl_row_ptr", row_ptr)
            self.register_buffer("excl_col_indices", col_indices)
        else:
            self.exclusions_coo = None
            self.excl_row_ptr = None
            self.excl_col_indices = None

        if not use_customized_ops:
            all_pairs = torch.combinations(torch.arange(natoms), with_replacement=include_self)
            if exclusions is not None:
                all_pairs = all_pairs.to(device=exclusions.device)
                excl_indices = torch.min(exclusions, dim=1).values * natoms + torch.max(exclusions, dim=1).values
                all_pairs_indices = torch.min(all_pairs, dim=1).values * natoms + torch.max(all_pairs, dim=1).values
                mask = ~torch.isin(all_pairs_indices, excl_indices)
                all_pairs = all_pairs[mask]
            self.register_buffer("all_pairs", all_pairs)
            self.pbc = PBC()

    @classmethod
    def convert_pairs_coo_to_csr(
        cls,
        pairs_coo: torch.Tensor,
        num_atoms: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a COO pair list into CSR format for a symmetric matrix.

        Given a list of index pairs in COO (coordinate) format representing
        non-zero entries of a symmetric matrix, this method produces the
        equivalent CSR (Compressed Sparse Row) representation. Because the
        matrix is symmetric, both directions ``(i, j)`` and ``(j, i)`` are
        guaranteed in the output regardless of whether the input contains
        one or both directions.

        The conversion proceeds in four steps:

        1. **Symmetrize** -- concatenate ``pairs_coo`` with its
           column-flipped copy so that both ``(i, j)`` and ``(j, i)``
           are present.
        2. **Encode and deduplicate** -- map each pair to a scalar key
           ``row * num_atoms + col`` and call :func:`torch.unique` on
           the 1-D key tensor. This is more efficient than
           ``torch.unique(dim=0)`` and produces keys in sorted
           (row-major) order.
        3. **Decode** -- recover row and column indices from the unique
           keys via integer division and modulo.
        4. **Build** ``row_ptr`` -- count neighbors per row with
           :meth:`torch.Tensor.scatter_add_`, then take the cumulative
           sum to form the CSR indptr array.

        Parameters
        ----------
        pairs_coo : torch.Tensor
            Integer tensor of shape ``(N, 2)`` where each row ``[i, j]``
            denotes a non-zero entry. The input may contain only one
            direction of a symmetric pair, both directions, or a mix.
        num_atoms : int, optional
            Total number of atoms (i.e. the matrix dimension). When
            ``None`` (default), it is inferred as
            ``pairs_coo.max() + 1``. Specify this explicitly when
            isolated atoms with no pairs exist so that ``row_ptr`` has
            the correct length.

        Returns
        -------
        row_ptr : torch.Tensor
            Shape ``(num_atoms + 1,)``, dtype ``torch.long``.
            ``row_ptr[i]`` is the start index in ``col_indices`` for
            row *i*; ``row_ptr[i+1] - row_ptr[i]`` gives the number of
            neighbors of atom *i*.
        col_indices : torch.Tensor
            Shape ``(M,)``, dtype ``torch.long``. Column indices of all
            non-zero entries, sorted first by row then by column within
            each row. ``M`` is the total number of non-zero entries
            after symmetrization and deduplication.
        """
        all_pairs = torch.cat([pairs_coo, pairs_coo.flip(1)], dim=0)

        if num_atoms is None:
            num_atoms = all_pairs.max().item() + 1

        keys = all_pairs[:, 0].to(torch.long) * num_atoms + all_pairs[:, 1].to(torch.long)
        keys = torch.unique(keys)

        rows = keys // num_atoms
        col_indices = keys % num_atoms

        row_counts = torch.zeros(num_atoms, dtype=torch.long, device=pairs_coo.device)
        row_counts.scatter_add_(0, rows, torch.ones_like(rows))
        row_ptr = torch.zeros(num_atoms + 1, dtype=torch.long, device=pairs_coo.device)
        row_ptr[1:] = torch.cumsum(row_counts, dim=0)

        return row_ptr, col_indices

    def to(self, *args, **kwargs):
        model = super().to(*args, **kwargs)
        if hasattr(self, "all_pairs") and self.all_pairs is not None:
            self.all_pairs = self.all_pairs.to(torch.long)
        if hasattr(self, "exclusions_coo") and self.exclusions_coo is not None:
            self.exclusions_coo = self.exclusions_coo.to(torch.long)
        if hasattr(self, "excl_row_ptr") and self.excl_row_ptr is not None:
            self.excl_row_ptr = self.excl_row_ptr.to(torch.long)
        if hasattr(self, "excl_col_indices") and self.excl_col_indices is not None:
            self.excl_col_indices = self.excl_col_indices.to(torch.long)
        return model

    def _forward_python(self, coords: torch.Tensor, box: torch.Tensor, cutoff: float):
        drvecs = self.pbc(coords[self.all_pairs[:, 1]] - coords[self.all_pairs[:, 0]], box)
        dr = torch.norm(drvecs, dim=1)
        return self.all_pairs[dr < cutoff]

    def _forward_cpp(self, coords: torch.Tensor, box: torch.Tensor, cutoff: float, max_npairs: int = -1, padding: bool = False):
        if self.algorithm == "cell_list":
            if box is None:
                raise ValueError("Cell-list algorithm requires a periodic box")
            pairs, npairs = torch.ops.torchff.build_neighbor_list_cell_list(
                coords, box, cutoff, max_npairs,
                self.excl_row_ptr, self.excl_col_indices,
                self.include_self,
            )
        else:
            pairs, npairs = torch.ops.torchff.build_neighbor_list_nsquared(
                coords, box, cutoff, max_npairs,
                self.excl_row_ptr, self.excl_col_indices,
                self.include_self,
            )
        if padding:
            return pairs, npairs
        npairs_found = npairs.item()
        return pairs[:npairs_found]

    def forward(self, coords: torch.Tensor, box: torch.Tensor, cutoff: float, max_npairs: int = -1, padding: bool = False):
        if self.use_customized_ops:
            return self._forward_cpp(coords, box, cutoff, max_npairs, padding)
        else:
            if max_npairs != -1:
                warnings.warn(
                    "max_npairs is ignored when use_customized_ops=False; "
                    "the pure-Python path always evaluates all pairs.",
                    stacklevel=2,
                )
            if padding:
                warnings.warn(
                    "padding is ignored when use_customized_ops=False; "
                    "the pure-Python path always returns exact pairs.",
                    stacklevel=2,
                )
            return self._forward_python(coords, box, cutoff)




def build_neighbor_list_nsquared(
    coords: torch.Tensor, box: torch.Tensor,
    cutoff: float, max_npairs: int = -1, padding: bool = False,
    excl_row_ptr: Optional[torch.Tensor] = None,
    excl_col_indices: Optional[torch.Tensor] = None,
    include_self: bool = False,
    out: Optional[torch.Tensor] = None
):
    """Build a neighbor list using the O(N^2) algorithm.

    Parameters
    ----------
    coords : torch.Tensor
        Atom positions, shape ``(natoms, 3)``.
    box : torch.Tensor
        Periodic box vectors, shape ``(3, 3)``.
    cutoff : float
        Distance cutoff.
    max_npairs : int, optional
        Pre-allocated capacity for pairs. ``-1`` allocates for the
        worst case ``natoms*(natoms-1)/2``.
    padding : bool, optional
        If ``True``, return the full (possibly padded) pairs tensor
        instead of trimming to the actual count.
    excl_row_ptr : torch.Tensor, optional
        CSR row-pointer tensor of shape ``(natoms + 1,)`` for the
        exclusion list. Must be ``torch.long``.
    excl_col_indices : torch.Tensor, optional
        CSR column-indices tensor for the exclusion list. Must be
        ``torch.long``.  Pairs ``(i, j)`` present in this sparse
        structure are skipped during neighbor-list construction.
        Use :meth:`NeighborList.convert_pairs_coo_to_csr` to convert
        COO-format exclusion pairs into the required CSR tensors.
    include_self : bool, optional
        If ``True``, include self-pairs ``(i, i)``.
    out : torch.Tensor, optional
        Pre-allocated output tensor of shape ``(max_npairs, 2)``
        (dtype ``torch.long``).  When provided, the kernel writes
        into this tensor and ``max_npairs`` is ignored.

    Returns
    -------
    pairs : torch.Tensor
        Neighbor pairs, shape ``(npairs, 2)``.
    npairs : torch.Tensor
        Scalar tensor with the number of pairs found.
    """
    if out is not None:
        npairs = torch.ops.torchff.build_neighbor_list_nsquared_out(
            coords, box, cutoff, out, excl_row_ptr, excl_col_indices,
            include_self
        )
        pairs = out
    else:
        pairs, npairs = torch.ops.torchff.build_neighbor_list_nsquared(
            coords, box, cutoff, max_npairs, excl_row_ptr,
            excl_col_indices, include_self
        )
    if not padding:
        npairs_found = npairs.item()
        max_cap = pairs.size(0)
        if npairs_found > max_cap:
            raise RuntimeError(
                f"Too many neighbor pairs found. Maximum is {max_cap} "
                f"but found {npairs_found}"
            )
        pairs = pairs[:npairs_found]
    return pairs, npairs


def build_neighbor_list_cell_list(coords: torch.Tensor, box: torch.Tensor, cutoff: float, max_npairs: int = -1, cell_size: float = 0.4, padding: bool = False, shared: bool = False):
    if shared:
        return torch.ops.torchff.build_neighbor_list_cell_list_shared(coords, box, cutoff, max_npairs, cell_size, padding)
    else:
        return torch.ops.torchff.build_neighbor_list_cell_list(coords, box, cutoff, max_npairs, cell_size, padding)


def build_cluster_pairs(
    coords: torch.Tensor, box: torch.Tensor,
    cutoff: float, 
    exclusions: Optional[torch.Tensor] = None,
    cell_size: float = 0.4,
    max_num_interacting_clusters: int = -1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    (
        sorted_atom_indices,
        cluster_exclusions,
        bitmask_exclusions,
        interacting_clusters,
        interacting_atoms
    ) = torch.ops.torchff.build_cluster_pairs(
        coords,
        box,
        cutoff,
        exclusions,
        cell_size,
        max_num_interacting_clusters
    )
    return (
        sorted_atom_indices,
        cluster_exclusions,
        bitmask_exclusions,
        interacting_clusters,
        interacting_atoms
    )


def decode_cluster_pairs(
    coords: torch.Tensor, 
    box: torch.Tensor,
    sorted_atom_indices,
    cluster_exclusions,
    bitmask_exclusions,
    interacting_clusters,
    interacting_atoms,
    cutoff: float,
    max_npairs: int = -1,
    padding: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torchff.decode_cluster_pairs(
        coords, 
        box,
        sorted_atom_indices,
        cluster_exclusions,
        bitmask_exclusions,
        interacting_clusters,
        interacting_atoms,
        cutoff,
        max_npairs,
        padding
    )


def build_neighbor_list_cluster_pairs(
    coords: torch.Tensor,
    box: torch.Tensor,
    cutoff: float,
    exclusions: Optional[torch.Tensor],
    cell_size: float = 0.45,
    max_num_interacting_clusters: int = -1,
    max_npairs: int = -1,
    padding: bool = False
):
    nblist = build_cluster_pairs(
        coords, box,
        cutoff, exclusions,
        cell_size, max_num_interacting_clusters
    )
    # sorted_atom_indices = nblist[0].detach().cpu().numpy().tolist()
    # print("Atom 137 in cluster", sorted_atom_indices.index(137)//32)
    # print("Atom 145 in cluster", sorted_atom_indices.index(145)//32)
    # for x, atoms in zip(nblist[-2].detach().cpu().numpy().tolist(), nblist[-1].detach().cpu().numpy().tolist()):
    #     print(x, atoms)
    # print(nblist[1].shape)
    for nl in nblist:
        print(nl.shape)
    print(nblist[-1][:5])
    return decode_cluster_pairs(
        coords, box, *nblist, cutoff,
        max_npairs, padding
    )



# def build_cluster_pairs(
#     coords: torch.Tensor, box: torch.Tensor,
#     cutoff: float, 
#     exclusions: Optional[torch.Tensor] = None,
#     cell_size: float = 0.4,
#     max_num_interacting_clusters: int = -1
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     if exclusions is None:
#         exclusions = torch.full((coords.shape[0], 1), -1, dtype=torch.int32, device=coords.device)
    
#     sorted_atom_indices, interacting_clusters, bitmask_exclusions, num_interacting_clusters = torch.ops.torchff.build_cluster_pairs(
#         coords,
#         box,
#         cutoff,
#         exclusions,
#         cell_size,
#         max_num_interacting_clusters
#     )
#     return (
#         sorted_atom_indices, interacting_clusters, 
#         bitmask_exclusions, num_interacting_clusters
#     )


# def decode_cluster_pairs(
#     coords: torch.Tensor, 
#     box: torch.Tensor,
#     sorted_atom_indices: torch.Tensor,
#     interacting_clusters: torch.Tensor,
#     bitmask_exclusions: torch.Tensor,
#     cutoff: float,
#     max_npairs: int = -1,
#     num_interacting_clusters: int = -1,
#     padding: bool = False
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     return torch.ops.torchff.decode_cluster_pairs(
#         coords, box, 
#         sorted_atom_indices, interacting_clusters, bitmask_exclusions,
#         cutoff, max_npairs, num_interacting_clusters, padding
#     )


# def build_neighbor_list_cluster_pairs(
#     coords: torch.Tensor,
#     box: torch.Tensor,
#     cutoff: float,
#     exclusions: Optional[torch.Tensor] = None,
#     cell_size: float = 0.45,
#     max_num_interacting_clusters: int = -1,
#     max_npairs: int = -1,
#     padding: bool = False
# ):
#     sorted_atom_indices, interacing_clusters, bitmask_exclusions, num_interacting_clusters = build_cluster_pairs(
#         coords, box,
#         cutoff, exclusions,
#         cell_size, max_num_interacting_clusters
#     )
#     # print(interacing_clusters)
#     # print(bitmask_exclusions)
#     # print("Found number of interacting clusters:", num_interacting_clusters.item())
#     return decode_cluster_pairs(
#         coords, box, sorted_atom_indices, interacing_clusters,
#         bitmask_exclusions, cutoff,
#         max_npairs, num_interacting_clusters.item(),
#         padding
#     )


