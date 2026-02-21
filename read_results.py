import os
import glob
import pickle
import re
import matplotlib.pyplot as plt

# --- CHANGE THIS to the folder containing your 5 pickle files ---
RESULTS_DIR = r"C:\VIT\Sem 6\Cryptography Lab\Project\apna\results\1771664737"

def plot_weight_distributions(directory):
    parsed_data = {}
    
    filepaths = glob.glob(os.path.join(directory, "*.pickle"))
    if not filepaths:
        print(f"No pickle files found in '{directory}'. Check your path!")
        return

    pattern = re.compile(r"_results-X(\d+)-N(\d+)-MAX_WEIGHT(\d+)")

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        match = pattern.search(filename)
        if match:
            m_val = int(match.group(1))
            n_val = int(match.group(2))
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                weights_dict = data[2] 
                
                if m_val not in parsed_data:
                    parsed_data[m_val] = {}
                parsed_data[m_val][n_val] = weights_dict
    
    if not parsed_data:
        print("Could not extract M and N values from the filenames.")
        return

    m_vals = sorted(parsed_data.keys())
    n_vals_set = set()
    for m in m_vals:
        n_vals_set.update(parsed_data[m].keys())
    n_vals = sorted(list(n_vals_set))

    rows = len(m_vals)
    cols = len(n_vals)

    # Increased base size per plot to give more breathing room
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for i, m in enumerate(m_vals):
        for j, n in enumerate(n_vals):
            ax = axes[i, j]
            
            if n in parsed_data[m]:
                w_dict = parsed_data[m][n]
                
                sorted_items = sorted(w_dict.items())
                weights = [item[0] for item in sorted_items]
                counts = [item[1] for item in sorted_items]
                
                total_weights = sum(counts)
                frequencies = [count / total_weights for count in counts]
                
                ax.bar(weights, frequencies, width=0.6)
                ax.set_title(f"TPM with M={m} and N={n}", fontsize=14, pad=10)
                
                # Increased Y-limit to 0.35 so tall edge bars aren't cut off
                ax.set_ylim(0, 0.35)
                
                ax.set_xticks(weights)
                # Changed rotation to 0 to keep the negative signs flat and readable
                ax.tick_params(axis='x', rotation=0) 
            else:
                ax.axis('off') 

    # Automatically adjusts padding to prevent text overlap
    plt.tight_layout() 
    
    print(f"Successfully loaded and plotted {len(filepaths)} files.")
    plt.show()

if __name__ == "__main__":
    plot_weight_distributions(RESULTS_DIR)