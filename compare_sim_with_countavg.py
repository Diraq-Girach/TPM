import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, deque

from tpm import TPM
from dynamic import DynamicTPM
from tools import signum_zero_to_plus, signum_zero_to_minus


def get_random_vec(max_x, k, n):
    """Helper function to generate non-zero inputs."""
    lst = list(range(-max_x, max_x + 1))
    if 0 in lst: lst.remove(0)
    return np.random.choice(lst, k * n)


def run_standard_tpm(k, n, l_max, max_x):
    """
    Compute tau_a and tau_b from the same pre-update weights,
    then apply both updates. Matches the original perform_key_agreement logic.
    """
    tpm_a = TPM(k, n, l_max, signum=signum_zero_to_plus)
    tpm_b = TPM(k, n, l_max, signum=signum_zero_to_minus)
    iterations = 0
    while not np.array_equal(tpm_a.weights, tpm_b.weights):
        iterations += 1
        vec = get_random_vec(max_x, k, n)

        # IMPORTANT: compute both taus before any optimize/update call
        tau_a, _ = tpm_a.get_output(vec)
        tau_b, _ = tpm_b.get_output(vec)

        tpm_a.optimize(vec, tau_b)
        tpm_b.optimize(vec, tau_a)

    return tpm_a.weights.flatten(), iterations


def run_diffused_hybrid_tpm(k, n, l_start, l_max, max_x,
                             threshold=0.92, window_size=30,
                             diffusion_steps=50, max_iterations=50000):
    """
    Hybrid Dynamic TPM with a safety guard against infinite loops.
    If the networks fail to sync within max_iterations, the run is aborted.
    """
    tpm_a = DynamicTPM(k, n, l_start, signum=signum_zero_to_plus)
    tpm_b = DynamicTPM(k, n, l_start, signum=signum_zero_to_minus)

    history = deque(maxlen=window_size)
    iterations, l_cur = 0, l_start
    is_synced = False
    diff_counter = 0
    clip_iters = 0
    mod_iters = 0

    while l_cur < l_max or not is_synced or diff_counter < diffusion_steps:

        # ── Safety guard ──────────────────────────────────────────────────
        if iterations >= max_iterations:
            print(f"  [WARN] Hit max_iterations={max_iterations} — aborting this run "
                  f"(l_cur={l_cur}, is_synced={is_synced}, diff_counter={diff_counter})")
            break
        # ─────────────────────────────────────────────────────────────────

        iterations += 1
        vec = get_random_vec(max_x, k, n)

        tau_a, _ = tpm_a.get_output(vec)
        tau_b, _ = tpm_b.get_output(vec)

        # Detect sync once L_max reached
        if not is_synced and l_cur == l_max:
            if np.array_equal(tpm_a.weights, tpm_b.weights):
                is_synced = True
                print(f"  Networks Synced at iteration {iterations}. Starting Diffusion...")

        if is_synced:
            diff_counter += 1

        is_final_stage = (l_cur == l_max)

        if is_final_stage:
            mod_iters += 1
        else:
            clip_iters += 1

        tpm_a.optimize(vec, tau_b, use_modulo=is_final_stage)
        tpm_b.optimize(vec, tau_a, use_modulo=is_final_stage)

        # Agreement-based L escalation trigger
        if l_cur < l_max:
            history.append(tau_a == tau_b)
            if len(history) == window_size and sum(history) / window_size >= threshold:
                new_l = l_cur + 1
                tpm_a.remap_weights(l_cur, new_l)
                tpm_b.remap_weights(l_cur, new_l)
                l_cur = new_l
                history.clear()

    return tpm_a.weights.flatten(), iterations, clip_iters, mod_iters, tpm_a.get_hashed_key()


def run_average(n_runs, k, n, l_max, l_start, max_x,
                threshold=0.92, window_size=30,
                diffusion_steps=50, max_iterations=50000):
    """
    Run both protocols n_runs times and aggregate results.
    Runs where the hybrid TPM hits max_iterations are flagged and excluded
    from averages so they don't skew the stats.
    """
    std_iters_all = []
    dyn_iters_all = []
    dyn_clip_all  = []
    dyn_mod_all   = []
    std_weights_all = []
    dyn_weights_all = []
    dyn_failed = 0

    for run_idx in range(1, n_runs + 1):
        print(f"[Run {run_idx}/{n_runs}] Standard TPM...")
        w_std, i_std = run_standard_tpm(k, n, l_max, max_x)
        std_iters_all.append(i_std)
        std_weights_all.extend(w_std.tolist())

        print(f"[Run {run_idx}/{n_runs}] Hybrid Dynamic TPM...")
        w_dyn, i_dyn, clip_iters, mod_iters, key = run_diffused_hybrid_tpm(
            k, n, l_start, l_max, max_x,
            threshold, window_size, diffusion_steps, max_iterations
        )

        # If we hit the guard, treat the run as failed and exclude from stats
        if i_dyn >= max_iterations:
            dyn_failed += 1
            print(f"  [SKIP] Run {run_idx} excluded from averages.")
        else:
            dyn_iters_all.append(i_dyn)
            dyn_clip_all.append(clip_iters)
            dyn_mod_all.append(mod_iters)
            dyn_weights_all.extend(w_dyn.tolist())

    print(f"\nHybrid TPM: {dyn_failed}/{n_runs} runs failed (hit max_iterations={max_iterations})")

    avg_std_iters  = np.mean(std_iters_all)
    std_std_iters  = np.std(std_iters_all)
    avg_dyn_iters  = np.mean(dyn_iters_all) if dyn_iters_all else float('nan')
    std_dyn_iters  = np.std(dyn_iters_all)  if dyn_iters_all else float('nan')
    avg_clip_iters = np.mean(dyn_clip_all)  if dyn_clip_all  else float('nan')
    avg_mod_iters  = np.mean(dyn_mod_all)   if dyn_mod_all   else float('nan')

    total_std_weights = len(std_weights_all)
    total_dyn_weights = len(dyn_weights_all) or 1  # avoid division by zero

    x_labels = list(range(-l_max, l_max + 1))
    s_freq = [Counter(std_weights_all).get(i, 0) / total_std_weights for i in x_labels]
    d_freq = [Counter(dyn_weights_all).get(i, 0) / total_dyn_weights for i in x_labels]

    return {
        "avg_std_iters":  avg_std_iters,
        "std_std_iters":  std_std_iters,
        "avg_dyn_iters":  avg_dyn_iters,
        "std_dyn_iters":  std_dyn_iters,
        "avg_clip_iters": avg_clip_iters,
        "avg_mod_iters":  avg_mod_iters,
        "dyn_failed":     dyn_failed,
        "n_runs":         n_runs,
        "x_labels":       x_labels,
        "s_freq":         s_freq,
        "d_freq":         d_freq,
        "std_iters_all":  std_iters_all,
        "dyn_iters_all":  dyn_iters_all,
    }


def plot_results(n_runs=1):
    K, N, MAX_X, L_MAX, L_START = 3, 100, 5, 5, 2

    if n_runs == 1:
        # ── Single-run mode (original behaviour) ──────────────────────────
        print("Simulating Standard Protocol (Clipped)...")
        w_std, i_std = run_standard_tpm(K, N, L_MAX, MAX_X)

        print("Simulating Hybrid Dynamic (Speed + Flatness)...")
        w_dyn, i_dyn, clip_iters, mod_iters, key = run_diffused_hybrid_tpm(
            K, N, L_START, L_MAX, MAX_X
        )
        print(f"\nFinal High-Entropy Key: {key}")

        x_labels = list(range(-L_MAX, L_MAX + 1))
        s_freq = [Counter(w_std.tolist()).get(i, 0) / len(w_std) for i in x_labels]
        d_freq = [Counter(w_dyn.tolist()).get(i, 0) / len(w_dyn) for i in x_labels]

        std_title = f"Standard TPM (Static L={L_MAX})\nSync: {i_std} iters"
        dyn_title = (
            f"Hybrid Dynamic TPM (Modulo Final)\n"
            f"Total: {i_dyn} | Clipped: {clip_iters} | Modulo: {mod_iters}"
        )

    else:
        # ── Multi-run averaging mode ───────────────────────────────────────
        print(f"Running {n_runs} simulations for each protocol...\n")
        stats = run_average(n_runs, K, N, L_MAX, L_START, MAX_X)

        x_labels = stats["x_labels"]
        s_freq   = stats["s_freq"]
        d_freq   = stats["d_freq"]

        std_title = (
            f"Standard TPM (Static L={L_MAX}) — avg over {n_runs} runs\n"
            f"Avg sync: {stats['avg_std_iters']:.1f} ± {stats['std_std_iters']:.1f} iters"
        )
        dyn_title = (
            f"Hybrid Dynamic TPM (Modulo Final) — avg over {n_runs} runs\n"
            f"Avg total: {stats['avg_dyn_iters']:.1f} ± {stats['std_dyn_iters']:.1f} | "
            f"Clipped: {stats['avg_clip_iters']:.1f} | Modulo: {stats['avg_mod_iters']:.1f}"
        )

        print(f"\n{'='*55}")
        print(f"Standard TPM  — avg iterations : {stats['avg_std_iters']:.1f} ± {stats['std_std_iters']:.1f}")
        print(f"Hybrid TPM    — avg iterations : {stats['avg_dyn_iters']:.1f} ± {stats['std_dyn_iters']:.1f}")
        print(f"Hybrid TPM    — failed runs    : {stats['dyn_failed']}/{stats['n_runs']}")
        print(f"{'='*55}")

    # ── Shared plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")

    ax[0].bar(x_labels, s_freq, color='steelblue')
    ax[0].set_title(std_title)
    ax[0].set_xlabel("Weight value")
    ax[0].set_ylabel("Frequency")

    ax[1].bar(x_labels, d_freq, color='forestgreen')
    ax[1].set_title(dyn_title)
    ax[1].set_xlabel("Weight value")

    for a in ax:
        a.set_ylim(0, 0.35)

    plt.suptitle(f"K={K}, N={N}, L_max={L_MAX}, max_x={MAX_X}", fontsize=12)
    plt.show()


if __name__ == "__main__":
    # Change n_runs to 1 to revert to original single-run behaviour
    plot_results(n_runs=10)