import os
import numpy as np

# --- CONFIGURATION ---
# Path to the folder containing "emg" and "Alignments" subfolders
BASE_PATH = "EMG-UKA-Trial-Corpus" 
OUTPUT_DIR = "./dataset_words"
# Choose output format: 'npy' or 'csv'
SAVE_FORMAT = 'csv'

# EMG-UKA Constants
FS = 600             # Sampling Rate (Hz)
FRAME_SHIFT_MS = 10  # 10ms per frame
SAMPLES_PER_FRAME = int(FS * (FRAME_SHIFT_MS / 1000)) # = 6

def find_files(root_dir, prefix="", suffix=""):
    """Recursively finds files and returns a dict: { 'id': 'full_path' }"""
    file_map = {}
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.startswith(prefix) and f.endswith(suffix):
                # Extract unique ID by removing prefix and suffix
                clean_id = f
                if prefix: clean_id = clean_id.replace(prefix, "")
                if suffix: clean_id = clean_id.replace(suffix, "")
                file_map[clean_id] = os.path.join(root, f)
    return file_map

def parse_offset_file(filepath):
    """Reads EMG offsets from the second line: (Start_Sample, End_Sample)"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # The second line contains the EMG offsets [cite: 105]
        if len(lines) >= 2:
            parts = lines[1].strip().split()
            if len(parts) >= 2:
                # Convert to integer samples
                return int(float(parts[0])), int(float(parts[1]))
    return None

def parse_word_alignment(filepath):
    """Parses custom .txt alignment files (Frame_Start Frame_End Word)"""
    words = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            
            # Find the numeric start/end frames
            start, end, label = None, None, None
            for i in range(len(parts) - 2):
                try:
                    start_val = int(parts[i])
                    end_val = int(parts[i+1])
                    # Found the numbers, the label is everything after
                    label = "_".join(parts[i+2:]) 
                    start, end = start_val, end_val
                    break
                except ValueError:
                    continue
            
            if start is not None and label:
                words.append((start, end, label))
    return words

def load_emg(filepath):
    # Load binary data (16-bit short integer) [cite: 94]
    raw = np.fromfile(filepath, dtype=np.int16)
    # Reshape to (Time, Channels) and keep the first 6 muscle channels (discarding the 7th marker channel) [cite: 16, 95]
    return raw.reshape(-1, 7)[:, :6]

# --- MAIN EXECUTION ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Indexing files...")

# 1. Index all necessary file types
emg_files = find_files(BASE_PATH, prefix="e07_", suffix=".adc")
offset_files = find_files(BASE_PATH, prefix="offset_", suffix=".txt")
word_files = find_files(BASE_PATH, prefix="words_", suffix=".txt")

# 2. Intersection: We need ALL three files to process an utterance
common_ids = set(emg_files.keys()) & set(offset_files.keys()) & set(word_files.keys())
print(f"Found {len(common_ids)} complete sets (EMG + Offset + Alignment). Processing...")

count = 0
for fid in common_ids:
    try:
        # A. Load Raw EMG
        raw = load_emg(emg_files[fid])
        
        # B. Load Offset & Cut Signal (Synchronization)
        offsets = parse_offset_file(offset_files[fid])
        if not offsets: continue
        
        start_sample, end_sample = offsets
        
        # Cut the signal based on the offset file boundaries
        clean_signal = raw[start_sample:end_sample]
        
        # C. Load Words & Slice (Alignment)
        word_list = parse_word_alignment(word_files[fid])
        
        for i, (w_start_frame, w_end_frame, label) in enumerate(word_list):
            # Skip silent/noise markers
            if label in ["$", "SIL", "SP", "sil", "sp", "garbage"]: continue
            
            # Convert Frame -> Sample (relative to the clean_signal)
            cut_start = w_start_frame * SAMPLES_PER_FRAME
            cut_end = w_end_frame * SAMPLES_PER_FRAME
            
            # Bounds check
            if cut_end <= len(clean_signal):
                word_data = clean_signal[cut_start:cut_end]
                
                # Save
                fmt = SAVE_FORMAT.lower()
                if fmt == 'npy':
                    fname = f"{label}_{fid}_{i}.npy"
                    outpath = os.path.join(OUTPUT_DIR, fname)
                    np.save(outpath, word_data)
                elif fmt == 'csv':
                    fname = f"{label}_{fid}_{i}.csv"
                    outpath = os.path.join(OUTPUT_DIR, fname)
                    # If 1-D, save single column; if 2-D save rows x channels with header
                    if word_data.ndim == 1:
                        np.savetxt(outpath, word_data, delimiter=',', fmt='%d')
                    else:
                        # Save 2-D array without header: rows are samples, columns are channels
                        np.savetxt(outpath, word_data, delimiter=',', fmt='%d')
                else:
                    # Unknown format: fallback to .npy
                    fname = f"{label}_{fid}_{i}.npy"
                    outpath = os.path.join(OUTPUT_DIR, fname)
                    np.save(outpath, word_data)
                count += 1
                
    except Exception as e:
        print(f"Error on {fid}: {e}")

print(f"Done! Successfully extracted {count} isolated words to '{OUTPUT_DIR}'")