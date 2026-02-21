import os
import glob
import pickle
import re
import matplotlib.pyplot as plt

# --- CHANGE THIS to the folder containing your pickle files ---
RESULTS_DIR = r"results\1771666247"

def plot_weight_distributions(directory,
                              per_plot_size=(4.0, 3.0),   # (width, height) in inches per subplot
                              max_ylim=0.40,
                              bar_width=0.4,
                              left=0.08, right=0.98, bottom=0.06, top=0.95,
                              wspace=0.35, hspace=0.45):
    """
    Balanced layout:
      - per_plot_size: width,height (inches) for each subplot. Increase height to make bars look taller.
      - max_ylim: y-axis top (0..1) to avoid chopping top of bars.
      - bar_width: width of bars.
      - subplots_adjust parameters (left,right,bottom,top,wspace,hspace) tune spacing.
    """
    parsed_data = {}

    filepaths = glob.glob(os.path.join(directory, "*.pickle"))
    if not filepaths:
        print(f"No pickle files found in '{directory}'. Check your path!")
        return

    pattern = re.compile(r"_results-X(\d+)-N(\d+)-MAX_WEIGHT(\d+)")

    for filepath in sorted(filepaths):
        filename = os.path.basename(filepath)
        match = pattern.search(filename)
        if match:
            m_val = int(match.group(1))
            n_val = int(match.group(2))

            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                # expected structure: data[2] is weights dict (weight -> count)
                weights_dict = data[2]

                parsed_data.setdefault(m_val, {})[n_val] = weights_dict

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

    # Compute figure size so each subplot roughly has the desired size
    per_w, per_h = per_plot_size
    figsize = (max(1.0, cols * per_w), max(1.0, rows * per_h))

    # Create subplots (do NOT force box aspect)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, squeeze=False)

    # Tweak font sizes depending on grid size to avoid overlaps
    scale = max(1, max(rows, cols) / 3.0)
    title_fs = max(9, int(12 / scale))
    tick_fs = max(7, int(9 / scale))

    for i, m in enumerate(m_vals):
        for j, n in enumerate(n_vals):
            ax = axes[i, j]

            if n in parsed_data[m]:
                w_dict = parsed_data[m][n]

                if not w_dict:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                sorted_items = sorted(w_dict.items())
                weights = [item[0] for item in sorted_items]
                counts = [item[1] for item in sorted_items]

                total_weights = sum(counts) if sum(counts) > 0 else 1
                frequencies = [count / total_weights for count in counts]

                ax.bar(weights, frequencies, width=bar_width)
                ax.set_title(f"TPM M={m} N={n}", fontsize=title_fs, pad=4)

                ax.set_ylim(0, max_ylim)

                ax.set_xticks(weights)
                ax.tick_params(axis='x', rotation=0, labelsize=tick_fs)
                ax.tick_params(axis='y', labelsize=tick_fs)

                # Minor grid to emphasize height of bars (optional)
                ax.yaxis.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
            else:
                ax.axis('off')

    # Balanced spacing — avoids tight_layout (which can clip) and also avoids forcing axes shape
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace, hspace=hspace)

    print(f"Successfully loaded and plotted {len(filepaths)} files (grid {rows}x{cols}).")
    plt.show()


if __name__ == "__main__":
    # For your 5 rows x 2 cols example, (per_plot_size=(4.0,3.0)) yields figsize ~ (8,15)
    # Increase the height slightly if you still want bars to appear taller, e.g. (4.0, 3.5).
    plot_weight_distributions(RESULTS_DIR,
                              per_plot_size=(4.0, 3.0),
                              max_ylim=0.40,
                              bar_width=0.4,
                              left=0.08, right=0.98, bottom=0.06, top=0.95,
                              wspace=0.35, hspace=0.45)