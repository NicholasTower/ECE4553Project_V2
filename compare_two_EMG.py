import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# --- CONFIGURATION ---
# Path to the folder where the previous script saved the .npy files
DATA_DIR = "./dataset_words" 

# The words you want to compare
TARGET_WORDS = ["PEOPLE", "NUMBER"] 

# Number of instances to plot for each word
NUM_INSTANCES = 2

FS = 600 # Sampling Rate (Hz)

def find_and_load_data(target_words, num_instances, data_folder):
    """Finds the required samples, loads them, and stores them with their info."""
    plot_data = []
    
    for word in target_words:
        # Search for files matching the word pattern (e.g., "THE_*.npy")
        search_pattern = os.path.join(data_folder, f"{word}_*.npy")
        matching_files = glob.glob(search_pattern)

        if len(matching_files) < num_instances:
            print(f"Warning: Only found {len(matching_files)} instances of '{word}'. Skipping this word.")
            continue
            
        # Select the required number of instances
        for i in range(num_instances):
            file_path = matching_files[i]
            data = np.load(file_path)
            
            # Convert samples to time (seconds)
            time_axis = np.arange(data.shape[0]) / FS
            
            plot_data.append({
                'word': word,
                'instance_num': i + 1,
                'path': os.path.basename(file_path),
                'data': data,
                'time': time_axis
            })
            
    return plot_data

def generate_comparison_plot(plot_data):
    """Generates the 2x2 grid plot for comparison."""
    if len(plot_data) != 4:
        print("Error: Could not collect exactly 4 samples for comparison.")
        return

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    # Flatten the 2x2 array of axes into a 1D list for easier iteration
    axes = axes.flatten() 
    
    fig.suptitle(f"EMG Comparison: 2 Instances of 'THE' vs. 2 Instances of 'BY'", fontsize=16, y=1.02)
    
    for i, item in enumerate(plot_data):
        ax = axes[i]
        
        # Determine unique label for the subplot title
        title = f"{item['word']} - Instance {item['instance_num']}"
        title += f"\nDuration: {item['time'][-1]:.2f}s | Shape: {item['data'].shape}"
        ax.set_title(title)
        
        # Plot all 6 channels with a vertical offset
        for channel in range(item['data'].shape[1]):
            # Use a fixed offset for clarity
            offset = channel * 500 
            ax.plot(item['time'], item['data'][:, channel] + offset, label=f"Ch {channel+1}") 

        # Style the plot
        ax.set_xlabel("Time (Seconds)")
        ax.set_ylabel("EMG Amplitude (Offset)")
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- EXECUTION ---
comparison_data = find_and_load_data(TARGET_WORDS, NUM_INSTANCES, DATA_DIR)
generate_comparison_plot(comparison_data)