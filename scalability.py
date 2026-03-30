"""
scalability.py
──────────────
Scalability comparison: Binary TPM vs Non-binary TPM vs Hybrid Dynamic TPM.

SCENARIO
────────
N users want to establish pairwise shared keys.  They need to run C(N, 2)
independent TPM synchronisations.  As N grows, the total synchronisation
cost grows as O(N²).  This file measures:

  • Avg iterations per pair   — raw sync speed of each protocol
  • Std deviation per pair    — consistency / reliability
  • Total iterations for the group  (avg/pair × C(N,2)) — overall cost
  • Per-pair failure rate     — robustness under the max_iterations guard

PROTOCOLS
─────────
  1. Binary TPM       inputs ∈ {-1, +1},  weights clipped to [-L, L]
  2. Non-binary TPM   inputs ∈ {-X,..,+X}\\{0},  weights clipped to [-L, L]
  3. Hybrid Dynamic   starts at L_start, escalates to L_max (modulo final stage)

Binary TPM == Non-binary TPM with max_x = 1, but kept separate for clarity.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from tpm import TPM
from dynamic import DynamicTPM
from tools import signum_zero_to_plus, signum_zero_to_minus


# ── Input generators ──────────────────────────────────────────────────────────

def binary_vec(k, n):
    """Strict binary inputs: {-1, +1}."""
    return np.random.choice([-1, 1], k * n)


def nonbinary_vec(max_x, k, n):
    """Non-binary inputs: {-max_x, …, -1, 1, …, max_x}."""
    lst = list(range(-max_x, max_x + 1))
    lst.remove(0)
    return np.random.choice(lst, k * n)


# ── Per-pair sync functions ───────────────────────────────────────────────────

def sync_binary_tpm(k, n, l, max_iterations=100_000):
    """
    Standard TPM with binary inputs.
    Both taus are computed from pre-update weights (consistent with the
    perform_key_agreement logic in main.py).
    """
    tpm_a = TPM(k, n, l, signum=signum_zero_to_plus)
    tpm_b = TPM(k, n, l, signum=signum_zero_to_minus)

    for i in range(1, max_iterations + 1):
        vec = binary_vec(k, n)
        tau_a, _ = tpm_a.get_output(vec)
        tau_b, _ = tpm_b.get_output(vec)
        tpm_a.optimize(vec, tau_b)
        tpm_b.optimize(vec, tau_a)
        if np.array_equal(tpm_a.weights, tpm_b.weights):
            return i, True

    return max_iterations, False


def sync_nonbinary_tpm(k, n, l, max_x, max_iterations=100_000):
    """Standard TPM with non-binary inputs and clipping."""
    tpm_a = TPM(k, n, l, signum=signum_zero_to_plus)
    tpm_b = TPM(k, n, l, signum=signum_zero_to_minus)

    for i in range(1, max_iterations + 1):
        vec = nonbinary_vec(max_x, k, n)
        tau_a, _ = tpm_a.get_output(vec)
        tau_b, _ = tpm_b.get_output(vec)
        tpm_a.optimize(vec, tau_b)
        tpm_b.optimize(vec, tau_a)
        if np.array_equal(tpm_a.weights, tpm_b.weights):
            return i, True

    return max_iterations, False


def sync_hybrid_tpm(k, n, l_start, l_max, max_x,
                    threshold=0.92, window_size=30,
                    diffusion_steps=50, max_iterations=100_000):
    """Hybrid Dynamic TPM: clipping during escalation, modulo at final L."""
    tpm_a = DynamicTPM(k, n, l_start, signum=signum_zero_to_plus)
    tpm_b = DynamicTPM(k, n, l_start, signum=signum_zero_to_minus)

    history      = deque(maxlen=window_size)
    l_cur        = l_start
    is_synced    = False
    diff_counter = 0

    for i in range(1, max_iterations + 1):
        vec = nonbinary_vec(max_x, k, n)
        tau_a, _ = tpm_a.get_output(vec)
        tau_b, _ = tpm_b.get_output(vec)

        if not is_synced and l_cur == l_max:
            if np.array_equal(tpm_a.weights, tpm_b.weights):
                is_synced = True

        if is_synced:
            diff_counter += 1

        is_final = (l_cur == l_max)
        tpm_a.optimize(vec, tau_b, use_modulo=is_final)
        tpm_b.optimize(vec, tau_a, use_modulo=is_final)

        if l_cur < l_max:
            history.append(tau_a == tau_b)
            if len(history) == window_size and sum(history) / window_size >= threshold:
                new_l = l_cur + 1
                tpm_a.remap_weights(l_cur, new_l)
                tpm_b.remap_weights(l_cur, new_l)
                l_cur = new_l
                history.clear()

        if l_cur == l_max and is_synced and diff_counter >= diffusion_steps:
            return i, True

    return max_iterations, False


# ── Multi-user simulation ─────────────────────────────────────────────────────

def simulate_group(n_users, sync_fn):
    """
    Simulate all C(n_users, 2) pairwise key agreements for a group of n_users.
    Each pair is independent (parallel model).

    Returns
    -------
    iters_list  : list[int]   — iterations for each pair
    success_list: list[bool]  — whether each pair synced before max_iterations
    """
    pairs = list(itertools.combinations(range(n_users), 2))
    iters_list   = []
    success_list = []
    for _ in pairs:
        iters, success = sync_fn()
        iters_list.append(iters)
        success_list.append(success)
    return iters_list, success_list


# ── Main experiment ───────────────────────────────────────────────────────────

def run_scalability(user_counts, n_repeats,
                    k, n, l, l_start, l_max, max_x):
    """
    For each (protocol, N_users) pair, run n_repeats full group exchanges and
    aggregate statistics across all pairs and all repeats.

    Parameters
    ----------
    user_counts : list[int]   Values of N (number of users) to test.
    n_repeats   : int         How many independent group rounds to average.
    k, n, l     : int         TPM shape and weight range.
    l_start     : int         Hybrid starting weight range.
    l_max       : int         Hybrid final weight range (== l for fair comparison).
    max_x       : int         Non-binary / hybrid input range.
    """
    protocols = {
        'Binary TPM':     lambda: sync_binary_tpm(k, n, l),
        'Non-binary TPM': lambda: sync_nonbinary_tpm(k, n, l, max_x),
        'Hybrid Dynamic': lambda: sync_hybrid_tpm(k, n, l_start, l_max, max_x),
    }
    palette = {
        'Binary TPM':     'steelblue',
        'Non-binary TPM': 'forestgreen',
        'Hybrid Dynamic': 'darkorange',
    }

    # Storage: proto → metric → list (one entry per N in user_counts)
    agg = {proto: {'avg_pair': [], 'std_pair': [],
                   'total':    [], 'fail_pct': []}
           for proto in protocols}

    for n_users in user_counts:
        n_pairs = n_users * (n_users - 1) // 2
        print(f"\n── N={n_users} users  ({n_pairs} pairs) ──────────────────────────")

        for proto_name, sync_fn in protocols.items():
            all_iters   = []
            all_success = []

            for rep in range(1, n_repeats + 1):
                iters, successes = simulate_group(n_users, sync_fn)
                all_iters.extend(iters)
                all_success.extend(successes)

            avg   = np.mean(all_iters)
            std   = np.std(all_iters)
            # "total" is the cost if all pairs are serialised (worst-case model)
            total = avg * n_pairs
            fail  = 100.0 * (1 - np.mean(all_success))

            agg[proto_name]['avg_pair'].append(avg)
            agg[proto_name]['std_pair'].append(std)
            agg[proto_name]['total'].append(total)
            agg[proto_name]['fail_pct'].append(fail)

            print(f"  {proto_name:<20}  "
                  f"avg/pair = {avg:8.1f} ± {std:7.1f}  |  "
                  f"total = {total:10.1f}  |  "
                  f"fail = {fail:5.1f}%")

    return agg, palette


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_scalability(user_counts, agg, palette, k, n, l, l_max, max_x):
    n_pairs_axis = [u * (u - 1) // 2 for u in user_counts]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    fig.suptitle(
        f"TPM Scalability: Binary vs Non-binary vs Hybrid Dynamic\n"
        f"K={k}, N={n}, L={l}, max_x={max_x}  —  pairwise key agreement",
        fontsize=13
    )

    # ── Plot 1: Avg iterations per pair ──────────────────────────────────
    ax = axes[0]
    for proto, data in agg.items():
        ax.errorbar(user_counts, data['avg_pair'], yerr=data['std_pair'],
                    marker='o', capsize=4, label=proto, color=palette[proto],
                    linewidth=1.8, markersize=6)
    ax.set_title("Avg Iterations per Pair\n(sync speed per user pair)")
    ax.set_xlabel("Number of Users (N)")
    ax.set_ylabel("Iterations")
    ax.set_xticks(user_counts)
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    # ── Plot 2: Total group cost ──────────────────────────────────────────
    ax = axes[1]
    for proto, data in agg.items():
        ax.plot(user_counts, data['total'],
                marker='s', label=proto, color=palette[proto],
                linewidth=1.8, markersize=6)

    # Annotate with C(N,2) on a secondary x-axis label
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(user_counts)
    ax2.set_xticklabels([f"C({u},2)={p}" for u, p in zip(user_counts, n_pairs_axis)],
                        fontsize=7, rotation=20)
    ax2.set_xlabel("Pair count", fontsize=8)

    ax.set_title("Total Group Iterations\n(avg/pair × C(N,2) pairs — sequential cost)")
    ax.set_xlabel("Number of Users (N)")
    ax.set_ylabel("Total Iterations")
    ax.set_xticks(user_counts)
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    # ── Plot 3: Failure rate ──────────────────────────────────────────────
    ax = axes[2]
    for proto, data in agg.items():
        ax.plot(user_counts, data['fail_pct'],
                marker='^', label=proto, color=palette[proto],
                linewidth=1.8, markersize=6)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title("Per-Pair Failure Rate\n(% of pairs hitting max_iterations)")
    ax.set_xlabel("Number of Users (N)")
    ax.set_ylabel("Failure Rate (%)")
    ax.set_xticks(user_counts)
    ax.set_ylim(-1, max(5, ax.get_ylim()[1]))
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.show()


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(user_counts, agg):
    print(f"\n{'='*75}")
    print("  SCALABILITY SUMMARY")
    print(f"{'='*75}")
    header = f"  {'Protocol':<20} {'N':>4}  {'avg/pair':>10}  {'±std':>8}  {'total':>12}  {'fail%':>7}"
    print(header)
    print(f"  {'-'*70}")
    for proto, data in agg.items():
        for i, n_users in enumerate(user_counts):
            print(f"  {proto:<20} {n_users:>4}  "
                  f"{data['avg_pair'][i]:>10.1f}  "
                  f"{data['std_pair'][i]:>8.1f}  "
                  f"{data['total'][i]:>12.1f}  "
                  f"{data['fail_pct'][i]:>7.2f}%")
        print(f"  {'-'*70}")
    print(f"{'='*75}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── TPM configuration ─────────────────────────────────────────────────
    K       = 3     # hidden units
    N       = 100   # neurons per hidden unit
    L       = 5     # weight range [-L, L] for binary / non-binary
    L_START = 2     # hybrid starting range
    L_MAX   = 5     # hybrid final range (matches L for fair comparison)
    MAX_X   = 5     # input range for non-binary / hybrid

    # ── Experiment configuration ──────────────────────────────────────────
    # USER_COUNTS: N values to sweep; keep small to stay tractable.
    # Pairs grow as C(N,2): N=2→1, N=4→6, N=6→15, N=8→28, N=10→45
    USER_COUNTS = [2, 4, 6, 8, 10]
    N_REPEATS   = 3     # independent group rounds per (protocol, N)

    agg, palette = run_scalability(
        USER_COUNTS, N_REPEATS,
        K, N, L, L_START, L_MAX, MAX_X
    )

    print_summary(USER_COUNTS, agg)
    plot_scalability(USER_COUNTS, agg, palette, K, N, L, L_MAX, MAX_X)