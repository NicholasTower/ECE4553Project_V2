import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# --- CONFIGURATION ---
# Path to the folder where the previous script saved the .npy files
DATA_DIR = "./dataset_words" 

# The word you want to compare (case-sensitive to the labels in words_*.txt)
TARGET_WORD = "THIS" 

FS = 600 # Sampling Rate (Hz)

def compare_two_instances(target_word, data_folder):
    """
    Finds the first two saved instances of the target_word and plots them 
    side-by-side to show intraclass variability.
    """
    # 1. Search for files matching the pattern (e.g., "THE_*.npy")
    search_pattern = os.path.join(data_folder, f"{target_word}_*.npy")
    matching_files = glob.glob(search_pattern)

    if len(matching_files) < 5:
        print(f"ERROR: Found only {len(matching_files)} instances of '{target_word}'.")
        print("Please run the full processing script again to generate more data.")
        return

    # Select the first two instances found
    file1_path = matching_files[0]
    file2_path = matching_files[1]
    file3_path = matching_files[2]
    file4_path = matching_files[3]
    file5_path = matching_files[4]

    # 2. Load data
    data1 = np.load(file1_path)
    data2 = np.load(file2_path)
    data3 = np.load(file3_path)
    data4 = np.load(file4_path)

    # 3. Prepare plot axes (convert samples to seconds)
    time_axis1 = np.arange(data1.shape[0]) / FS
    time_axis2 = np.arange(data2.shape[0]) / FS

    # 4. Create figure with two subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=False)
    fig.suptitle(f"Comparison of Two Instances of the Word: '{target_word}'", fontsize=16)

    # Plot 1: First Instance
    axes[0].set_title(f"Instance 1: ({os.path.basename(file1_path)}) - Duration: {time_axis1[-1]:.2f}s")
    # Plot channels with vertical offset
    for channel in range(data1.shape[1]):
        axes[0].plot(time_axis1, data1[:, channel] + (channel * 500), label=f"Ch {channel+1}") 
    axes[0].set_ylabel("EMG Amplitude (Offset)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True)
    
    # Plot 2: Second Instance
    axes[1].set_title(f"Instance 2: ({os.path.basename(file2_path)}) - Duration: {time_axis2[-1]:.2f}s")
    for channel in range(data2.shape[1]):
        axes[1].plot(time_axis2, data2[:, channel] + (channel * 500), label=f"Ch {channel+1}") 
    axes[1].set_xlabel("Time (Seconds)")
    axes[1].set_ylabel("EMG Amplitude (Offset)")
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    plt.show()

# Run the comparison
compare_two_instances(TARGET_WORD, DATA_DIR)