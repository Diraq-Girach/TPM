import numpy as np
import matplotlib.pyplot as plt
from collections import deque, Counter

from dynamic import DynamicTPM
from tools import signum_zero_to_plus, signum_zero_to_minus


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_random_vec(max_x, k, n):
    lst = list(range(-max_x, max_x + 1))
    if 0 in lst:
        lst.remove(0)
    return np.random.choice(lst, k * n)


def hex_to_bits(hex_string):
    """Convert a hex digest string to a binary bit string."""
    return bin(int(hex_string, 16))[2:].zfill(len(hex_string) * 4)


def shannon_entropy(bit_string):
    """Shannon entropy in bits per symbol for a binary string."""
    if not bit_string:
        return 0.0
    counts = Counter(bit_string)
    total  = len(bit_string)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())


def bit_balance(bit_string):
    """Fraction of 1s in a bit string (ideal = 0.5)."""
    if not bit_string:
        return float('nan')
    return bit_string.count('1') / len(bit_string)


# ── Core simulation ───────────────────────────────────────────────────────────

def run_hybrid_tpm(k, n, l_start, l_max, max_x,
                   threshold=0.92, window_size=30,
                   diffusion_steps=50, max_iterations=50000):
    tpm_a = DynamicTPM(k, n, l_start, signum=signum_zero_to_plus)
    tpm_b = DynamicTPM(k, n, l_start, signum=signum_zero_to_minus)

    history      = deque(maxlen=window_size)
    iterations   = 0
    l_cur        = l_start
    is_synced    = False
    diff_counter = 0

    while l_cur < l_max or not is_synced or diff_counter < diffusion_steps:

        if iterations >= max_iterations:
            print(f"  [WARN] max_iterations={max_iterations} reached — run aborted.")
            return None, iterations

        iterations += 1
        vec = get_random_vec(max_x, k, n)

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

    return tpm_a, iterations


# ── Multi-run comparison ──────────────────────────────────────────────────────

def compare_extraction_methods(n_runs=10, k=3, n=100, l_start=2,
                                l_max=5, max_x=5):
    results = {
        "iterations"     : [],

        # SHA-256 metrics
        "sha_keys"       : [],
        "sha_bit_lengths": [],
        "sha_entropies"  : [],
        "sha_balances"   : [],

        # Von Neumann metrics
        "vn_keys"        : [],
        "vn_raw_bits"    : [],
        "vn_bit_lengths" : [],
        "vn_entropies"   : [],
        "vn_balances"    : [],

        "failed"         : 0,
    }

    for run_idx in range(1, n_runs + 1):
        print(f"[Run {run_idx}/{n_runs}]", end=" ", flush=True)
        tpm_a, iters = run_hybrid_tpm(k, n, l_start, l_max, max_x)

        if tpm_a is None:
            results["failed"] += 1
            print("FAILED — skipped.")
            continue

        # ── SHA-256 extraction ────────────────────────────────────────────
        sha_key      = tpm_a.get_hashed_key()
        sha_bits     = hex_to_bits(sha_key)          # 256-bit string
        sha_entropy  = shannon_entropy(sha_bits)
        sha_balance  = bit_balance(sha_bits)

        # ── Von Neumann extraction ────────────────────────────────────────
        vn_key, raw_bits = tpm_a.get_von_neumann_key()
        vn_entropy   = shannon_entropy(raw_bits)
        vn_balance   = bit_balance(raw_bits)

        results["iterations"].append(iters)

        results["sha_keys"].append(sha_key)
        results["sha_bit_lengths"].append(len(sha_bits))
        results["sha_entropies"].append(sha_entropy)
        results["sha_balances"].append(sha_balance)

        results["vn_keys"].append(vn_key)
        results["vn_raw_bits"].append(raw_bits)
        results["vn_bit_lengths"].append(len(raw_bits))
        results["vn_entropies"].append(vn_entropy)
        results["vn_balances"].append(vn_balance)

        print(
            f"iters: {iters:5d} | "
            f"SHA entropy: {sha_entropy:.4f} bal: {sha_balance:.3f} | "
            f"VN bits: {len(raw_bits):4d} entropy: {vn_entropy:.4f} bal: {vn_balance:.3f}"
        )

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary(results, n_runs):
    good = n_runs - results["failed"]
    print(f"\n{'='*65}")
    print(f"  Runs completed : {good}/{n_runs}  ({results['failed']} failed)")

    if not results["iterations"]:
        print("  No successful runs to summarise.")
        print(f"{'='*65}")
        return

    print(f"  Avg iterations : {np.mean(results['iterations']):.1f} "
          f"± {np.std(results['iterations']):.1f}")

    print(f"\n  {'Metric':<30} {'SHA-256':>15} {'Von Neumann':>15}")
    print(f"  {'-'*60}")

    sha_len_mean = np.mean(results["sha_bit_lengths"])
    vn_len_mean  = np.mean(results["vn_bit_lengths"])
    print(f"  {'Avg bit length':<30} {sha_len_mean:>15.1f} {vn_len_mean:>15.1f}")

    sha_ent_mean = np.mean(results["sha_entropies"])
    vn_ent_mean  = np.mean(results["vn_entropies"])
    sha_ent_std  = np.std(results["sha_entropies"])
    vn_ent_std   = np.std(results["vn_entropies"])
    print(f"  {'Avg entropy (ideal=1.0)':<30} "
          f"{sha_ent_mean:>10.4f}±{sha_ent_std:.4f} "
          f"{vn_ent_mean:>10.4f}±{vn_ent_std:.4f}")

    sha_bal_mean = np.mean(results["sha_balances"])
    vn_bal_mean  = np.mean(results["vn_balances"])
    sha_bal_std  = np.std(results["sha_balances"])
    vn_bal_std   = np.std(results["vn_balances"])
    print(f"  {'Avg bit balance (ideal=0.5)':<30} "
          f"{sha_bal_mean:>10.4f}±{sha_bal_std:.4f} "
          f"{vn_bal_mean:>10.4f}±{vn_bal_std:.4f}")

    unique_sha = len(set(results["sha_keys"]))
    unique_vn  = len(set(results["vn_keys"]))
    print(f"  {'Unique keys':<30} {unique_sha:>15} {unique_vn:>15}")
    print(f"{'='*65}\n")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(results, n_runs):
    if not results["iterations"]:
        print("Nothing to plot.")
        return

    good       = n_runs - results["failed"]
    run_labels = list(range(1, good + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle(
        f"SHA-256 vs Von Neumann Key Extraction — Hybrid Dynamic TPM\n"
        f"({good}/{n_runs} successful runs)",
        fontsize=13
    )

    # ── Plot 1: Bit length per run ────────────────────────────────────────
    ax = axes[0, 0]
    width = 0.35
    x = np.arange(len(run_labels))
    ax.bar(x - width/2, results["sha_bit_lengths"],
           width, color="steelblue", label="SHA-256 (fixed 256 bits)")
    ax.bar(x + width/2, results["vn_bit_lengths"],
           width, color="darkorange", label="Von Neumann (variable)")
    ax.set_title("Extracted Bit Length per Run")
    ax.set_xlabel("Run")
    ax.set_ylabel("Bits")
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels)
    ax.legend()

    # ── Plot 2: Entropy per run ───────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(run_labels, results["sha_entropies"],
            marker='o', color="steelblue", label="SHA-256")
    ax.plot(run_labels, results["vn_entropies"],
            marker='s', color="darkorange", label="Von Neumann")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Ideal (1.0)")
    ax.set_title("Shannon Entropy per Run")
    ax.set_xlabel("Run")
    ax.set_ylabel("Entropy (bits/symbol)")
    ax.set_ylim(0, 1.15)
    ax.legend()

    # ── Plot 3: Bit balance per run ───────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(run_labels, results["sha_balances"],
            marker='o', color="steelblue", label="SHA-256")
    ax.plot(run_labels, results["vn_balances"],
            marker='s', color="darkorange", label="Von Neumann")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Ideal (0.5)")
    ax.set_title("Bit Balance (fraction of 1s) per Run")
    ax.set_xlabel("Run")
    ax.set_ylabel("Fraction of 1-bits")
    ax.set_ylim(0, 1.0)
    ax.legend()

    # ── Plot 4: Avg metric bar chart (summary) ────────────────────────────
    ax = axes[1, 1]
    metrics      = ["Entropy", "Bit Balance"]
    sha_means    = [np.mean(results["sha_entropies"]),
                    np.mean(results["sha_balances"])]
    vn_means     = [np.mean(results["vn_entropies"]),
                    np.mean(results["vn_balances"])]
    sha_stds     = [np.std(results["sha_entropies"]),
                    np.std(results["sha_balances"])]
    vn_stds      = [np.std(results["vn_entropies"]),
                    np.std(results["vn_balances"])]
    ideals       = [1.0, 0.5]

    x = np.arange(len(metrics))
    bars_sha = ax.bar(x - width/2, sha_means, width,
                      yerr=sha_stds, capsize=5,
                      color="steelblue", label="SHA-256")
    bars_vn  = ax.bar(x + width/2, vn_means, width,
                      yerr=vn_stds, capsize=5,
                      color="darkorange", label="Von Neumann")
    for i, ideal in enumerate(ideals):
        ax.plot([i - width, i + width], [ideal, ideal],
                color="red", linestyle="--", linewidth=1.5)

    ax.set_title("Avg Entropy & Bit Balance (± std)\nRed dashes = ideal")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.2)
    ax.legend()

    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N_RUNS = 10
    results = compare_extraction_methods(n_runs=N_RUNS)
    print_summary(results, N_RUNS)
    plot_comparison(results, N_RUNS)