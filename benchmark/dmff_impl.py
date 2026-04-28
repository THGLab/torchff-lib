import time
import numpy as np
import jax
import jax.numpy as jnp
from dmff.admp.pme import generate_pme_recip, Ck_1, pme_self
from dmff.admp.multipole import convert_cart2harm


def perf_jax(func, *args, desc="perf_jax", warmup=10, repeat=1000):
    """Benchmark a JAX function with proper GPU synchronization.

    JAX uses async dispatch, so we call jax.block_until_ready on the result
    to ensure computation completes before measuring elapsed time.
    """
    for _ in range(warmup):
        out = func(*args)
        jax.block_until_ready(out)

    # perf = []
    # for _ in range(repeat):
    #     start = time.perf_counter()
    #     out = func(*args)
    #     jax.block_until_ready(out)
    #     end = time.perf_counter()
    #     perf.append((end - start) * 1000)

    start = time.perf_counter()
    for _ in range(repeat):
        out = func(*args)
    jax.block_until_ready(out)
    end = time.perf_counter()
    perf = [(end - start) * 1000 / repeat for _ in range(repeat)]

    perf = np.array(perf)
    print(f"{desc} - Time: {np.mean(perf):.4f} +/- {np.std(perf):.4f} ms")
    return perf


def prepare_dmff_multipoles(q, p, t, rank):
    """Pack cartesian multipoles and convert to spherical harmonics for DMFF PME.

    Parameters
    ----------
    q : jnp.ndarray
        Charges, shape (N,).
    p : jnp.ndarray or None
        Dipoles, shape (N, 3). Required if rank >= 1.
    t : jnp.ndarray or None
        Quadrupoles (symmetric traceless), shape (N, 3, 3). Required if rank >= 2.
    rank : int
        Multipole rank: 0 (charge), 1 (dipole), 2 (quadrupole).

    Returns
    -------
    jnp.ndarray
        Spherical harmonic multipoles Q, shape (N, (rank+1)**2).
    """
    N = q.shape[0]
    if rank == 0:
        Theta = jnp.reshape(q, (N, 1))
    elif rank == 1:
        Theta = jnp.concatenate([jnp.reshape(q, (N, 1)), p], axis=1)
    else:
        # rank == 2: pack quadrupole as [xx, xy, xz, yy, yz, zz]
        t_flat = jnp.stack([
            t[:, 0, 0], t[:, 0, 1], t[:, 0, 2],
            t[:, 1, 1], t[:, 1, 2], t[:, 2, 2]
        ], axis=1)
        Theta = jnp.concatenate([
            jnp.reshape(q, (N, 1)), p, t_flat
        ], axis=1)
    lmax = rank
    Q = convert_cart2harm(Theta, lmax)
    return Q


def generate_compute_pme_recip_dmff(alpha, kmax, rank):

    recip_fn = generate_pme_recip(
        Ck_1, alpha, False, 6, kmax, kmax, kmax, rank
    )

    def compute_pme_recip_dmff(coords, box, Q):
        return recip_fn(coords, box, Q) + pme_self(Q, alpha, rank)

    return jax.jit(jax.value_and_grad(compute_pme_recip_dmff, argnums=(0, 2)))
