"""
fair_compare_extraction.py
──────────────────────────
Fair comparison of SHA-256 vs Von Neumann + SHA-256 key extraction.


WHAT THIS FILE DOES
────────────────────
Metrics collected on the *pre-hash* data:
  • Shannon entropy  (ideal = 1.0 bits/symbol)
  • Bit balance / fraction of 1s  (ideal = 0.5)
  • VN extraction yield  (bits_out / bits_in, theoretical max ≈ 0.5)
  • Final key uniqueness across runs  (collision resistance proxy)

The weight sign-bits going into Path A are the *same* bits that enter the VN
extractor in Path B, so the comparison is on equal footing.
"""

import hashlib
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, deque

from dynamic import DynamicTPM
from tools import signum_zero_to_plus, signum_zero_to_minus


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_random_vec(max_x, k, n):
    lst = list(range(-max_x, max_x + 1))
    lst.remove(0)
    return np.random.choice(lst, k * n)


def weights_to_sign_bits(weights) -> str:
    """
    Flatten weights → sign-bit string, dropping exact zeros.

    A weight > 0  maps to '1'; weight < 0 maps to '0'; weight == 0 is dropped
    (carries no directional information and would inflate the input length
    while contributing no entropy).
    """
    flat = weights.flatten()
    return ''.join('1' if w > 0 else '0' for w in flat if w != 0)


def von_neumann_extract(bit_string: str) -> str:
    """
    Von Neumann randomness extractor (de-biasing step).

    Process consecutive non-overlapping pairs:
      (0, 1)  →  emit '1'
      (1, 0)  →  emit '0'
      (0, 0) or (1, 1)  →  discard  (correlated / biased pair)

    The output is unbiased even if the input has a fixed per-symbol bias,
    provided successive pairs are i.i.d.  The yield approaches H(p)(1-H(p))
    where p is the bias — for a fair coin (p=0.5) the theoretical max is 0.5.
    """
    extracted = []
    i = 0
    while i + 1 < len(bit_string):
        a, b = bit_string[i], bit_string[i + 1]
        if a != b:
            extracted.append(b)
        i += 2  # always consume the pair
    return ''.join(extracted)


def sha256_of_bits(bit_string: str) -> str:
    """Pack a bit string into bytes (right-padded with 0s) and SHA-256 hash it."""
    if not bit_string:
        return hashlib.sha256(b'\x00').hexdigest()
    # Pad to the nearest multiple of 8
    padded = bit_string.ljust((len(bit_string) + 7) // 8 * 8, '0')
    data = bytes(int(padded[i:i + 8], 2) for i in range(0, len(padded), 8))
    return hashlib.sha256(data).hexdigest()


def shannon_entropy(bit_string: str) -> float:
    """Shannon entropy in bits/symbol for a binary string."""
    if not bit_string:
        return 0.0
    counts = Counter(bit_string)
    total = len(bit_string)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())


def bit_balance(bit_string: str) -> float:
    """Fraction of 1-bits (ideal = 0.5)."""
    if not bit_string:
        return float('nan')
    return bit_string.count('1') / len(bit_string)


# ── TPM runner ────────────────────────────────────────────────────────────────

def run_hybrid_tpm(k, n, l_start, l_max, max_x,
                   threshold=0.92, window_size=30,
                   diffusion_steps=50, max_iterations=50_000):
    tpm_a = DynamicTPM(k, n, l_start, signum=signum_zero_to_plus)
    tpm_b = DynamicTPM(k, n, l_start, signum=signum_zero_to_minus)

    history = deque(maxlen=window_size)
    iterations = 0
    l_cur = l_start
    is_synced = False
    diff_counter = 0

    while l_cur < l_max or not is_synced or diff_counter < diffusion_steps:
        if iterations >= max_iterations:
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


# ── Core comparison ───────────────────────────────────────────────────────────

def compare_fair(n_runs=10, k=3, n=100, l_start=2, l_max=5, max_x=5):
    """
    For each successful run, extract the synced weight matrix and compute:

    Path A (SHA only):
      1. Convert weights → sign-bit string  (pre-hash input)
      2. Measure entropy & balance of that string
      3. SHA-256(sign-bits)  →  final key

    Path B (VN + SHA):
      1. Same sign-bit string  (same starting point as Path A)
      2. Apply Von Neumann extractor  →  de-biased bits
      3. Measure entropy & balance of extracted bits
      4. Measure VN yield (extracted / input)
      5. SHA-256(extracted-bits)  →  final key
    """
    results = {
        'iterations':   [],
        # Pre-hash quality for Path A
        'raw_entropy':  [],
        'raw_balance':  [],
        'raw_lengths':  [],
        # Pre-hash quality for Path B
        'vn_entropy':   [],
        'vn_balance':   [],
        'vn_lengths':   [],
        'vn_yield':     [],
        # Final keys from both paths
        'sha_keys':     [],
        'vn_sha_keys':  [],
        'failed':       0,
    }

    for run_idx in range(1, n_runs + 1):
        print(f"[Run {run_idx:>2}/{n_runs}]", end=" ", flush=True)
        tpm_a, iters = run_hybrid_tpm(k, n, l_start, l_max, max_x)

        if tpm_a is None:
            results['failed'] += 1
            print("FAILED (hit max_iterations) — skipped.")
            continue

        # ── Shared starting point: weight sign-bits ───────────────────────
        raw_bits = weights_to_sign_bits(tpm_a.weights)

        # ── Path A metrics ────────────────────────────────────────────────
        raw_ent = shannon_entropy(raw_bits)
        raw_bal = bit_balance(raw_bits)
        sha_key = sha256_of_bits(raw_bits)

        # ── Path B metrics ────────────────────────────────────────────────
        vn_bits  = von_neumann_extract(raw_bits)
        vn_ent   = shannon_entropy(vn_bits)
        vn_bal   = bit_balance(vn_bits)
        vn_yield = len(vn_bits) / len(raw_bits) if raw_bits else 0.0
        vn_sha_key = sha256_of_bits(vn_bits)

        # ── Store ─────────────────────────────────────────────────────────
        results['iterations'].append(iters)
        results['raw_entropy'].append(raw_ent)
        results['raw_balance'].append(raw_bal)
        results['raw_lengths'].append(len(raw_bits))
        results['vn_entropy'].append(vn_ent)
        results['vn_balance'].append(vn_bal)
        results['vn_lengths'].append(len(vn_bits))
        results['vn_yield'].append(vn_yield)
        results['sha_keys'].append(sha_key)
        results['vn_sha_keys'].append(vn_sha_key)

        print(
            f"iters={iters:5d} | "
            f"Raw({len(raw_bits)}b): ent={raw_ent:.4f} bal={raw_bal:.3f} | "
            f"VN({len(vn_bits)}b): ent={vn_ent:.4f} bal={vn_bal:.3f} "
            f"yield={vn_yield:.3f}"
        )

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary(results, n_runs):
    good = n_runs - results['failed']
    print(f"\n{'='*72}")
    print(f"  Runs completed  : {good}/{n_runs}  ({results['failed']} failed)")

    if not results['iterations']:
        print("  No successful runs to summarise.")
        print(f"{'='*72}\n")
        return

    print(f"  Avg iterations  : {np.mean(results['iterations']):.1f} "
          f"± {np.std(results['iterations']):.1f}")
    print()
    print(f"  {'Metric':<40} {'Path A: SHA':>15} {'Path B: VN+SHA':>15}")
    print(f"  {'-'*70}")

    def row(label, a_vals, b_vals, fmt=".4f"):
        print(f"  {label:<40} "
              f"{np.mean(a_vals):>10{fmt}}±{np.std(a_vals):{fmt}} "
              f"{np.mean(b_vals):>10{fmt}}±{np.std(b_vals):{fmt}}")

    row("Pre-hash entropy   (ideal = 1.0)", results['raw_entropy'], results['vn_entropy'])
    row("Pre-hash bit balance (ideal = 0.5)", results['raw_balance'], results['vn_balance'])
    row("Pre-hash bit length", results['raw_lengths'], results['vn_lengths'], fmt=".1f")

    print(f"\n  {'VN yield (bits_out / bits_in)':<40} "
          f"{'N/A':>15} "
          f"{np.mean(results['vn_yield']):>10.4f}±{np.std(results['vn_yield']):.4f}")

    unique_sha = len(set(results['sha_keys']))
    unique_vn  = len(set(results['vn_sha_keys']))
    print(f"  {'Unique final keys':<40} {unique_sha:>15} {unique_vn:>15}")
    print(f"{'='*72}\n")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results, n_runs):
    if not results['iterations']:
        print("Nothing to plot.")
        return

    good = n_runs - results['failed']
    runs = list(range(1, good + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle(
        "Fair Key Extraction Comparison: SHA-256 vs Von Neumann + SHA-256\n"
        "Measuring pre-hash data quality — both paths end with SHA-256",
        fontsize=13
    )

    # ── Plot 1: Pre-hash entropy ──────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(runs, results['raw_entropy'],
            marker='o', color='steelblue', label='Path A: raw sign-bits → SHA-256')
    ax.plot(runs, results['vn_entropy'],
            marker='s', color='darkorange', label='Path B: VN bits → SHA-256')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='Ideal (1.0)')
    ax.set_title("Pre-Hash Shannon Entropy per Run")
    ax.set_xlabel("Run")
    ax.set_ylabel("Entropy (bits/symbol)")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)

    # ── Plot 2: Pre-hash bit balance ──────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(runs, results['raw_balance'],
            marker='o', color='steelblue', label='Path A: raw sign-bits')
    ax.plot(runs, results['vn_balance'],
            marker='s', color='darkorange', label='Path B: VN bits')
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Ideal (0.5)')
    ax.set_title("Pre-Hash Bit Balance per Run")
    ax.set_xlabel("Run")
    ax.set_ylabel("Fraction of 1-bits")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)

    # ── Plot 3: VN extraction yield ───────────────────────────────────────
    ax = axes[1, 0]
    ax.bar(runs, results['vn_yield'], color='darkorange', label='VN yield')
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1,
               label='Theoretical max ≈ 0.5 (fair coin)')
    ax.set_title("Von Neumann Yield per Run\n(bits extracted / bits input)")
    ax.set_xlabel("Run")
    ax.set_ylabel("Yield")
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=8)

    # ── Plot 4: Summary bar (avg ± std) ───────────────────────────────────
    ax = axes[1, 1]
    width = 0.35
    metrics  = ["Entropy", "Bit Balance"]
    raw_means = [np.mean(results['raw_entropy']), np.mean(results['raw_balance'])]
    vn_means  = [np.mean(results['vn_entropy']),  np.mean(results['vn_balance'])]
    raw_stds  = [np.std(results['raw_entropy']),  np.std(results['raw_balance'])]
    vn_stds   = [np.std(results['vn_entropy']),   np.std(results['vn_balance'])]
    ideals    = [1.0, 0.5]

    x = np.arange(len(metrics))
    ax.bar(x - width / 2, raw_means, width, yerr=raw_stds, capsize=5,
           color='steelblue', label='Path A: SHA-256 only')
    ax.bar(x + width / 2, vn_means, width, yerr=vn_stds, capsize=5,
           color='darkorange', label='Path B: VN + SHA-256')
    for i, ideal in enumerate(ideals):
        ax.plot([i - width, i + width], [ideal, ideal],
                'r--', linewidth=1.5)

    ax.set_title("Avg Pre-Hash Quality (± std)\nRed dashes = ideal target")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=8)

    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N_RUNS = 10
    results = compare_fair(n_runs=N_RUNS)
    print_summary(results, N_RUNS)
    plot_results(results, N_RUNS)