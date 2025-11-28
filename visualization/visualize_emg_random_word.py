import numpy as np
import matplotlib.pyplot as plt
import os
import random

# ----------------- CONFIG -----------------
# 1. Update this to the folder where your .npy files were saved
DATA_DIR = "./dataset_words" 
FS = 600 # The sampling rate used to collect the data

def visualize_random_word(data_folder):
    """Loads a random .npy file and plots its 6 channels."""
    all_files = [f for f in os.listdir(data_folder) if f.endswith(".npy")]

    if not all_files:
        print(f"Error: No .npy files found in {data_folder}. Check the DATA_DIR path.")
        return

    # Choose a random file and load it
    choice = random.choice(all_files)
    data = np.load(os.path.join(data_folder, choice))
    
    # Extract info from filename (e.g., "THIS_002_001_0100_0.npy" -> "THIS")
    word_name = choice.split("_")[0] 
    
    time_samples = data.shape[0]
    time_axis = np.arange(time_samples) / FS # Convert samples to seconds

    # Create the plot
    plt.figure(figsize=(12, 6))
    num_channels = data.shape[1]

    # Plot each channel with a vertical offset for clarity
    for channel in range(num_channels):
        # Vertical offset for stacking the lines
        offset = channel * 500 
        plt.plot(time_axis, data[:, channel] + offset, label=f"Ch {channel+1}") 

    plt.title(f"EMG Signal for Isolated Word: '{word_name}' | Shape: {data.shape}")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Amplitude (Vertical Offset Applied)")
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

# Run the function
visualize_random_word(DATA_DIR)