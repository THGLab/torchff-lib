#pragma once

// Linearly search for x in list[0..n-1], skipping entries equal to skip_val.
template <typename index_t>
__device__ __forceinline__ bool in_list(
    index_t x, const index_t* list, index_t n, index_t skip_val
) {
    if (!list) {
        return false;
    }
    bool found = false;
    for (index_t pos = 0; pos < n; ++pos) {
        if (list[pos] == skip_val) {
            continue;
        }
        if (x == list[pos]) {
            found = true;
            break;
        }
    }
    return found;
}

// Check if (atom_i, atom_j) is excluded using CSR-format exclusion lists.
// row_ptr[i]..row_ptr[i+1] gives the range of col_indices belonging to atom i.
__device__ __forceinline__ bool is_excluded_csr(
    int64_t atom_i, int64_t atom_j,
    const int64_t* row_ptr, const int64_t* col_indices
) {
    if (!col_indices) return false;
    int64_t start = row_ptr[atom_i];
    int64_t end   = row_ptr[atom_i + 1];
    for (int64_t e = start; e < end; ++e) {
        if (col_indices[e] == atom_j) return true;
    }
    return false;
}
