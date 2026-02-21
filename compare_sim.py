import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, deque

# Import original classes and tools
from tpm import TPM
from dynamic import DynamicTPM
from tools import signum_zero_to_plus, signum_zero_to_minus

def get_random_vec(max_x, k, n):
    """Helper function to generate non-binary inputs."""
    lst = list(range(-max_x, max_x + 1))
    if 0 in lst: lst.remove(0)
    return np.random.choice(lst, k * n)

def run_standard_tpm(k, n, l_max, max_x):
    tpm_a = TPM(k, n, l_max, signum=signum_zero_to_plus)
    tpm_b = TPM(k, n, l_max, signum=signum_zero_to_minus)
    iterations = 0
    while not np.array_equal(tpm_a.weights, tpm_b.weights):
        iterations += 1
        vec = get_random_vec(max_x, k, n)
        tpm_a.optimize(vec, tpm_b.get_output(vec)[0])
        tpm_b.optimize(vec, tpm_a.get_output(vec)[0])
    return tpm_a.weights.flatten(), iterations

def run_diffused_hybrid_tpm(k, n, l_start, l_max, max_x, threshold=0.92, window_size=30, diffusion_steps=150):
    tpm_a = DynamicTPM(k, n, l_start, signum=signum_zero_to_plus)
    tpm_b = DynamicTPM(k, n, l_start, signum=signum_zero_to_minus)
    
    history = deque(maxlen=window_size)
    iterations, l_cur = 0, l_start
    is_synced = False
    diff_counter = 0
    
    # Loop continues until L_max is hit, weights match, AND diffusion is complete
    while l_cur < l_max or not is_synced or diff_counter < diffusion_steps:
        iterations += 1
        vec = get_random_vec(max_x, k, n)
        
        tau_a, _ = tpm_a.get_output(vec)
        tau_b, _ = tpm_b.get_output(vec)
        
        # Check if synchronization has been achieved
        if not is_synced and l_cur == l_max:
            if np.array_equal(tpm_a.weights, tpm_b.weights):
                is_synced = True
                print(f"Networks Synced at iteration {iterations}. Starting Diffusion...")

        # If already synced, increment the diffusion counter
        if is_synced:
            diff_counter += 1

        # Final stage (L=max) always uses Modulo for flatness
        is_final_stage = (l_cur == l_max)
        tpm_a.optimize(vec, tau_b, use_modulo=is_final_stage)
        tpm_b.optimize(vec, tau_a, use_modulo=is_final_stage)
        
        # Agreement-Based Trigger for speed
        if l_cur < l_max:
            history.append(tau_a == tau_b)
            if len(history) == window_size and sum(history)/window_size >= threshold:
                new_l = l_cur + 1
                tpm_a.remap_weights(l_cur, new_l)
                tpm_b.remap_weights(l_cur, new_l)
                l_cur = new_l
                history.clear()
            
    return tpm_a.weights.flatten(), iterations, tpm_a.get_hashed_key()

def plot_results():
    K, N, MAX_X, L_MAX, L_START = 3, 1000, 5, 5, 2
    
    print("Simulating Standard Protocol (Clipped)...")
    w_std, i_std = run_standard_tpm(K, N, L_MAX, MAX_X)
    
    print("Simulating Hybrid Dynamic (Speed + Flatness)...")
    w_dyn, i_dyn, key = run_diffused_hybrid_tpm(K, N, L_START, L_MAX, MAX_X)
    
    print(f"\nFinal High-Entropy Key: {key}")

    x_labels = list(range(-L_MAX, L_MAX + 1))
    s_freq = [Counter(w_std.tolist()).get(i, 0) / len(w_std) for i in x_labels]
    d_freq = [Counter(w_dyn.tolist()).get(i, 0) / len(w_dyn) for i in x_labels]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    ax[0].bar(x_labels, s_freq, color='steelblue')
    ax[0].set_title(f"Standard TPM (Static L=5)\nSync: {i_std} iters")
    
    ax[1].bar(x_labels, d_freq, color='forestgreen')
    ax[1].set_title(f"Hybrid Dynamic TPM (Modulo Final)\nSync: {i_dyn} iters")
    
    for a in ax: a.set_ylim(0, 0.35)
    plt.show()

if __name__ == "__main__":
    plot_results()