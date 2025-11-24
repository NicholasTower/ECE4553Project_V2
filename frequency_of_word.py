import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
import glob



# --- CONFIGURATION ---
DATA_DIR = "./dataset_words" # Directory containing your processed .npy files
TARGET_WORD = "THIS"                # The word you want to analyze (must exist in DATA_DIR)
FS = 600                           # Sampling rate in Hz

def plot_word_psd_subplots(data_folder, target_word, fs):
    """Loads a processed word sample, calculates PSD, and plots each channel."""
    
    # 1. Search for a file matching the target word
    search_pattern = os.path.join(data_folder, f"{target_word}_*.npy")
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        print(f"❌ Error: No samples found for word '{target_word}' in '{data_folder}'.")
        print("Please ensure the word is capitalized and the processing script ran successfully.")
        return

    # Load the first instance found
    file_path = matching_files[0]
    data = np.load(file_path) # Shape: (Time Samples, 6 Channels)
    
    time_samples, num_channels = data.shape
    
    print(f"Analyzing file: {os.path.basename(file_path)}")
    print(f"Sample length: {time_samples} samples ({time_samples/fs:.2f} seconds)")

    # 2. Create Subplots
    fig, axes = plt.subplots(nrows=num_channels, ncols=1, figsize=(10, 15), sharex=True)
    fig.suptitle(f'PSD of Isolated Word: "{target_word}"{file_path}', fontsize=16)

    # 3. Analyze and Plot Each Channel
    for i in range(num_channels):
        signal = data[:, i]
        
        # Calculate PSD using Welch's method. nperseg must be less than the sample length.
        nperseg = min(512, time_samples // 2) 
        frequencies, psd = welch(signal, fs=fs, nperseg=nperseg)
        
        # Convert to dB for clearer visualization
        psd_db = 10 * np.log10(psd)
        
        # Plot
        axes[i].plot(frequencies, psd_db, color='tab:purple')
        
        # Diagnostic Lines (Tune filters based on these):
        axes[i].axvline(x=50, color='r', linestyle='--', alpha=0.7)
        axes[i].axvspan(20, 300, color='green', alpha=0.1)

        axes[i].set_ylabel(f"Ch {i+1} (dB)")
        axes[i].grid(True, linestyle='--', alpha=0.6)

    axes[0].legend(['Signal', '50Hz Line Noise', 'Target Band'], loc='upper right')
    axes[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

# Run the visualization
plot_word_psd_subplots(DATA_DIR, TARGET_WORD, FS)