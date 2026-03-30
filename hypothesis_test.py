"""
hypothesis_test.py
──────────────────
Validates the correlation hypothesis of the Hybrid TPM weights.
Executes:
  1. Lag-1 Autocorrelation (Pre-hash)
  2. Wald-Wolfowitz Runs Test (Pre-hash)
  3. Pairwise Hamming Distance (Post-hash)
"""

import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from dynamic import DynamicTPM
from tools import signum_zero_to_plus, signum_zero_to_minus
import compare_extraction_better as fce  # Assuming the previous script was saved as this

def lag1_autocorrelation(bit_string: str) -> float:
    """Pearson r between b[i] and b[i+1] for all i."""
    bits = [int(b) for b in bit_string]
    if len(bits) < 2:
        return float('nan')
    
    x = np.array(bits[:-1], dtype=float)
    y = np.array(bits[1:], dtype=float)
    
    x -= x.mean()
    y -= y.mean()
    
    denom = np.sqrt((x**2).sum() * (y**2).sum())
    return float(np.dot(x, y) / denom) if denom > 0 else 0.0

def wald_wolfowitz_runs_test(bit_string: str):
    """
    Standard Wald-Wolfowitz Runs Test for sequence randomness.
    Returns (Z-statistic, p-value).
    """
    bits = [int(b) for b in bit_string]
    n = len(bits)
    if n == 0:
        return 0.0, 1.0

    n1 = sum(bits)
    n0 = n - n1
    
    # Count actual runs
    runs = 1
    for i in range(1, n):
        if bits[i] != bits[i-1]:
            runs += 1
            
    # Expected runs and variance
    expected_runs = ((2 * n0 * n1) / n) + 1
    variance = (2 * n0 * n1 * (2 * n0 * n1 - n)) / ((n ** 2) * (n - 1))
    
    if variance == 0:
        return 0.0, 1.0
        
    z_stat = (runs - expected_runs) / math.sqrt(variance)
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value

def hamming_distance(hex1: str, hex2: str) -> int:
    """Calculates the Hamming distance between two SHA-256 hex strings."""
    b1 = bin(int(hex1, 16))[2:].zfill(256)
    b2 = bin(int(hex2, 16))[2:].zfill(256)
    return sum(x != y for x, y in zip(b1, b2))

def run_hypothesis_tests(n_runs=50):
    print(f"Generating data for {n_runs} successful runs...\n")
    
    raw_bit_strings = []
    vn_bit_strings = []
    sha_keys = []
    vn_sha_keys = []
    
    successful_runs = 0
    while successful_runs < n_runs:
        # Using K=3, N=100, L=5 as standard test parameters
        tpm, _ = fce.run_hybrid_tpm(3, 100, 2, 5, 5)
        if tpm is None:
            continue
            
        raw_bits = fce.weights_to_sign_bits(tpm.weights)
        if not raw_bits:
            continue
            
        vn_bits = fce.von_neumann_extract(raw_bits)
        
        raw_bit_strings.append(raw_bits)
        vn_bit_strings.append(vn_bits)
        sha_keys.append(fce.sha256_of_bits(raw_bits))
        vn_sha_keys.append(fce.sha256_of_bits(vn_bits))
        
        successful_runs += 1
        print(f"\rCollected: {successful_runs}/{n_runs}", end="")
    print("\n\n── Pre-Hash Statistical Testing ──────────────────────────────")
    
    # 1. Autocorrelation
    raw_autocorr = [lag1_autocorrelation(b) for b in raw_bit_strings]
    mean_ac = np.mean(raw_autocorr)
    print(f"Lag-1 Autocorrelation (Raw bits): {mean_ac:.4f} ± {np.std(raw_autocorr):.4f}")
    if mean_ac > 0.05:
        print("  -> CONFIRMED: Noticeable positive correlation exists in adjacent weights.")
        print("  -> This directly explains the poor Von Neumann yield (~25%).")

    # 2. Runs Test
    raw_p_values = [wald_wolfowitz_runs_test(b)[1] for b in raw_bit_strings]
    raw_fails = sum(1 for p in raw_p_values if p < 0.05)
    print(f"\nWald-Wolfowitz Runs Test (alpha=0.05):")
    print(f"  Raw Bits Failed: {raw_fails}/{n_runs} ({(raw_fails/n_runs)*100:.1f}%)")
    
    print("\n── Post-Hash Avalanche Testing ───────────────────────────────")
    
    # 3. Hamming Distances
    sha_dists = [hamming_distance(a, b) for a, b in itertools.combinations(sha_keys, 2)]
    vn_dists = [hamming_distance(a, b) for a, b in itertools.combinations(vn_sha_keys, 2)]
    
    print(f"Expected Hamming Distance: 128.0 (for 256-bit hash)")
    print(f"SHA-256 only (Path A)    : {np.mean(sha_dists):.2f} ± {np.std(sha_dists):.2f}")
    print(f"VN + SHA-256 (Path B)    : {np.mean(vn_dists):.2f} ± {np.std(vn_dists):.2f}")
    
    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].hist(raw_p_values, bins=20, color='steelblue', alpha=0.7)
    ax[0].axvline(0.05, color='red', linestyle='--', label='alpha=0.05')
    ax[0].set_title("Pre-Hash Wald-Wolfowitz p-values")
    ax[0].set_xlabel("p-value (uniform is ideal)")
    ax[0].legend()
    
    ax[1].hist(sha_dists, bins=range(90, 160), alpha=0.6, label='Path A (SHA)', color='steelblue')
    ax[1].hist(vn_dists, bins=range(90, 160), alpha=0.6, label='Path B (VN+SHA)', color='darkorange')
    ax[1].axvline(128, color='red', linestyle='--', label='Ideal (128)')
    ax[1].set_title("Post-Hash Pairwise Hamming Distances")
    ax[1].set_xlabel("Hamming Distance (bits)")
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_hypothesis_tests(n_runs=100)