import csv
import math
import os
from typing import List, Dict, Any, Tuple, Optional
import traceback
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
torch.set_default_device('cuda')
import matplotlib.pyplot as plt

import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, '..'))
sys.path.insert(0, _script_dir)
from torchff.test_utils import perf_op
from torchff.pme import PME

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    gpu_devices = jax.devices("gpu")
    assert len(gpu_devices), "No GPU found!"
    if gpu_devices:
        jax.config.update("jax_default_device", gpu_devices[0])
    from dmff_impl import (
        generate_compute_pme_recip_dmff,
        prepare_dmff_multipoles,
        perf_jax,
    )
    RUN_DMFF = True
except Exception as e:
    print(f" Fail to import DMFF: {traceback.print_exc()}")
    RUN_DMFF = False



def create_pme_test_data(
    num_atoms: int,
    rank: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, int]:
    """Create random test data for PME benchmarks.

    Uses the same scaling and Ewald parameter heuristic as in
    :func:`tests.test_ewald.create_test_data`.
    """
    box_len = float((num_atoms * 10.0) ** (1.0 / 3.0))

    coords_np = np.random.rand(num_atoms, 3) * box_len
    q_np = np.random.randn(num_atoms)
    q_np -= q_np.mean()
    d_np = np.random.randn(num_atoms, 3)
    t_np = np.empty((num_atoms, 3, 3), dtype=float)
    for i in range(num_atoms):
        A = np.random.randn(3, 3)
        sym = 0.5 * (A + A.T)
        trace = np.trace(sym) / 3.0
        sym -= np.eye(3) * trace
        t_np[i] = sym

    box_np = np.eye(3) * box_len

    coords = torch.tensor(coords_np, device=device, dtype=dtype, requires_grad=True)
    box = torch.tensor(box_np, device=device, dtype=dtype)
    q = torch.tensor(q_np, device=device, dtype=dtype, requires_grad=True)
    p = (
        torch.tensor(d_np, device=device, dtype=dtype, requires_grad=True)
        # if rank >= 1
        # else None
    )
    t = (
        torch.tensor(t_np, device=device, dtype=dtype, requires_grad=True)
        # if rank >= 2
        # else None
    )

    alpha = math.sqrt(-math.log10(2.0 * 1e-6)) / 9.0
    max_hkl = 50
    for i in range(2, 50):
        error_estimate = (
            i * math.sqrt(box_len * alpha) / 20.0
        ) * math.exp(
            -math.pi * math.pi * i * i / (box_len * alpha * box_len * alpha)
        )
        if error_estimate < 1e-6:
            max_hkl = i
            break

    return coords, box, q, p, t, alpha, max_hkl


def run_single_benchmark(
    num_atoms: int,
    rank: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    repeat: int = 200,
) -> List[Dict[str, Any]]:
    """Run performance benchmark for PME at a given system size and multipole rank.

    Benchmarks TorchFF PME (Python ref and customized CUDA) and optionally
    TorchPME (charge-only, rank=0). Returns a list of result dicts.
    """
    results: List[Dict[str, Any]] = []

    coords, box, q, p, t, alpha, max_hkl = create_pme_test_data(
        num_atoms, rank, device=device, dtype=dtype
    )

    def _energy_from_pme(func, coords, box, q, p, t):
        out = func(coords, box, q, p, t)
        return out[3] if isinstance(out, tuple) else out

    # Reference: Python PME, no customized ops
    torch.cuda.empty_cache()
    try:
        func_ref = PME(alpha, max_hkl, rank, use_customized_ops=False).to(
                device=device, dtype=dtype
            )

        perf_ref = perf_op(
            lambda c, b, qq, pp, tt: _energy_from_pme(func_ref, c, b, qq, pp, tt),
            coords,
            box,
            q,
            p,
            t,
            desc=f"pme_ref (N={num_atoms}, rank={rank})",
            repeat=repeat,
            run_backward=True,
            use_cuda_graph=False,
        )
        results.append(
            {
                "rank": rank,
                "num_atoms": num_atoms,
                "impl": "ref",
                "mean_ms": float(np.mean(perf_ref)),
                "std_ms": float(np.std(perf_ref)),
                "success": True,
            }
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(
                f"Encountered OOM for ref implementation at "
                f"N={num_atoms}, rank={rank}: {e}"
            )
        else:
            print(
                f"Encountered RuntimeError for ref implementation at "
                f"N={num_atoms}, rank={rank}: {e}"
            )
        torch.cuda.empty_cache()
        results.append(
            {
                "rank": rank,
                "num_atoms": num_atoms,
                "impl": "ref",
                "mean_ms": float("nan"),
                "std_ms": float("nan"),
                "success": False,
            }
        )

    # Customized CUDA PME
    torch.cuda.empty_cache()
    try:
        func_custom = PME(
            alpha, max_hkl, rank, use_customized_ops=True
        ).to(device=device, dtype=dtype)
        perf_custom = perf_op(
            lambda c, b, qq, pp, tt: _energy_from_pme(func_custom, c, b, qq, pp, tt),
            coords,
            box,
            q,
            p,
            t,
            desc=f"pme_torchff (N={num_atoms}, rank={rank})",
            repeat=repeat,
            run_backward=True,
            use_cuda_graph=True,
            explicit_sync=False
        )
        results.append(
            {
                "rank": rank,
                "num_atoms": num_atoms,
                "impl": "custom",
                "mean_ms": float(np.mean(perf_custom)),
                "std_ms": float(np.std(perf_custom)),
                "success": True,
            }
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(
                f"Encountered OOM for customized implementation at "
                f"N={num_atoms}, rank={rank}: {e}"
            )
        else:
            print(
                f"Encountered RuntimeError for customized implementation at "
                f"N={num_atoms}, rank={rank}: {e}"
            )
        torch.cuda.empty_cache()
        results.append(
            {
                "rank": rank,
                "num_atoms": num_atoms,
                "impl": "custom",
                "mean_ms": float("nan"),
                "std_ms": float("nan"),
                "success": False,
            }
        )

    # TorchPME implementation (charge-only, rank=0)
    if rank == 0:
        torch.cuda.empty_cache()
        try:
            from torchpme_impl import PMETorchPME

            box_len = float(torch.mean(torch.diag(box)).item())
            # mesh_spacing = box_len / (2 * max_hkl + 1)
            mesh_spacing = box_len * 2 / (max_hkl - 1)
            charges = q.unsqueeze(1) if q.ndim == 1 else q
            func_torchpme = torch.compile(
                PMETorchPME(alpha=alpha, mesh=mesh_spacing).to(
                    device=device, dtype=dtype
                ), mode='max-autotune'
            )
            perf_torchpme = perf_op(
                func_torchpme,
                coords,
                box,
                charges,
                desc=f"pme_torchpme (N={num_atoms}, rank={rank})",
                repeat=repeat,
                run_backward=True,
                use_cuda_graph=False,
            )
            results.append(
                {
                    "rank": rank,
                    "num_atoms": num_atoms,
                    "impl": "torchpme",
                    "mean_ms": float(np.mean(perf_torchpme)),
                    "std_ms": float(np.std(perf_torchpme)),
                    "success": True,
                }
            )
        except ImportError as e:
            print(
                f"Skipping TorchPME benchmark (not installed or import failed): {e}"
            )
            results.append(
                {
                    "rank": rank,
                    "num_atoms": num_atoms,
                    "impl": "torchpme",
                    "mean_ms": float("nan"),
                    "std_ms": float("nan"),
                    "success": False,
                }
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(
                    f"Encountered OOM for TorchPME at "
                    f"N={num_atoms}, rank={rank}: {e}"
                )
            else:
                print(
                    f"Encountered RuntimeError for TorchPME at "
                    f"N={num_atoms}, rank={rank}: {e}"
                )
            torch.cuda.empty_cache()
            results.append(
                {
                    "rank": rank,
                    "num_atoms": num_atoms,
                    "impl": "torchpme",
                    "mean_ms": float("nan"),
                    "std_ms": float("nan"),
                    "success": False,
                }
            )
    
    # DMFF implementation (JAX)
    if RUN_DMFF:
        try:
            torch.cuda.empty_cache()
            coords_jax = jnp.array(coords.detach().cpu().numpy())
            box_jax = jnp.array(box.detach().cpu().numpy())
            q_jax = jnp.array(q.detach().cpu().numpy())
            p_jax = jnp.array(p.detach().cpu().numpy()) if p is not None else None
            t_jax = jnp.array(t.detach().cpu().numpy()) if t is not None else None

            Q_jax = prepare_dmff_multipoles(q_jax, p_jax, t_jax, rank)
            compute_fn = generate_compute_pme_recip_dmff(alpha, max_hkl, rank)

            perf_dmff = perf_jax(
                compute_fn,
                coords_jax,
                box_jax,
                Q_jax,
                desc=f"pme_dmff (N={num_atoms}, rank={rank})",
                warmup=10,
                repeat=repeat,
            )
            results.append(
                {
                    "rank": rank,
                    "num_atoms": num_atoms,
                    "impl": "dmff",
                    "mean_ms": float(np.mean(perf_dmff)),
                    "std_ms": float(np.std(perf_dmff)),
                    "success": True,
                }
            )
        except ImportError as e:
            print(
                f"Skipping DMFF benchmark (not installed or import failed): {e}"
            )
            results.append(
                {
                    "rank": rank,
                    "num_atoms": num_atoms,
                    "impl": "dmff",
                    "mean_ms": float("nan"),
                    "std_ms": float("nan"),
                    "success": False,
                }
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(
                    f"Encountered OOM for DMFF at "
                    f"N={num_atoms}, rank={rank}: {e}"
                )
            else:
                print(
                    f"Encountered RuntimeError for DMFF at "
                    f"N={num_atoms}, rank={rank}: {e}"
                )
            torch.cuda.empty_cache()
            results.append(
                {
                    "rank": rank,
                    "num_atoms": num_atoms,
                    "impl": "dmff",
                    "mean_ms": float("nan"),
                    "std_ms": float("nan"),
                    "success": False,
                }
            )

    return results


def write_results_to_csv(results: List[Dict[str, Any]], csv_path: str) -> None:
    """Write benchmark results to CSV."""
    fieldnames = ["rank", "num_atoms", "impl", "mean_ms", "std_ms", "success"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Wrote benchmark results to {csv_path}")


def plot_results(results: List[Dict[str, Any]], pdf_path: str) -> None:
    """Create a 3-panel PDF plot (one per rank) comparing PME implementations.

    Each subplot uses log-log scales: system size vs time (ms). Lines for
    ref, custom, and torchpme (rank=0 only).
    """
    ranks = [0, 1, 2]
    impl_labels = {
        "ref": "TorchFF (Python)",
        "custom": "TorchFF (CUDA)",
        "torchpme": "TorchPME",
        "dmff": "DMFF",
    }
    impl_styles = {"ref": "o-", "custom": "s-", "torchpme": "^-", "dmff": "d-"}
    impl_colors = {"ref": "C0", "custom": "C1", "torchpme": "C2", "dmff": "C3"}

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

    for ax, rank in zip(axes, ranks):
        rank_results = [r for r in results if r["rank"] == rank and r["success"]]
        impls_to_plot = ("ref", "custom", "torchpme", "dmff") if rank == 0 else ("ref", "custom", "dmff")
        for impl in impls_to_plot:
            impl_results = [r for r in rank_results if r["impl"] == impl]
            if not impl_results:
                continue
            impl_results.sort(key=lambda r: r["num_atoms"])
            sizes = [r["num_atoms"] for r in impl_results]
            mean_ms = [r["mean_ms"] for r in impl_results]
            ax.plot(
                sizes,
                mean_ms,
                impl_styles[impl],
                color=impl_colors[impl],
                label=impl_labels[impl],
                linewidth=1.5,
                markersize=5,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("System size (number of atoms)")
        ax.set_ylim(0.1, 300)
        ax.set_title(f"Rank {rank}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        if rank == 0:
            ax.set_ylabel("Time (ms)")
        ax.legend()

    plt.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved benchmark plot to {pdf_path}")


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = "cuda"
    dtype = torch.float64

    system_sizes = [100, 300, 1000, 3000, 10000, 30000, 100000]
    ranks = [0, 1, 2]

    all_results: List[Dict[str, Any]] = []

    for rank in ranks:
        for num_atoms in system_sizes:
            print(f"Running benchmark for N={num_atoms}, rank={rank}")
            try:
                bench_results = run_single_benchmark(
                    num_atoms=num_atoms,
                    rank=rank,
                    device=device,
                    dtype=dtype,
                    repeat=200,
                )
                all_results.extend(bench_results)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"Encountered OOM while setting up benchmark for "
                        f"N={num_atoms}, rank={rank}: {e}"
                    )
                else:
                    print(
                        f"Encountered RuntimeError while setting up benchmark for "
                        f"N={num_atoms}, rank={rank}: {e}"
                    )
                torch.cuda.empty_cache()
                impls = ("ref", "custom", "torchpme", "dmff") if rank == 0 else ("ref", "custom", "dmff")
                for impl in impls:
                    all_results.append(
                        {
                            "rank": rank,
                            "num_atoms": num_atoms,
                            "impl": impl,
                            "mean_ms": float("nan"),
                            "std_ms": float("nan"),
                            "success": False,
                        }
                    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "benchmark_pme.csv")
    pdf_path = os.path.join(script_dir, "benchmark_pme.pdf")

    write_results_to_csv(all_results, csv_path)
    plot_results(all_results, pdf_path)


if __name__ == "__main__":
    main()
