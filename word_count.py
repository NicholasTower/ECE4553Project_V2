import os
import glob
from collections import Counter

# --- Configuration ---
# ⚠️ IMPORTANT: Update this variable to the actual path where your files are stored.
DATA_DIR = 'dataset_words' # Assuming files are in the current directory

def count_and_sort_words(data_dir):
    """
    Scans the directory for .npy files, extracts the word from the filename, 
    and counts the occurrences, returning the results sorted by count (descending).
    """
    
    search_pattern = os.path.join(data_dir, '*.npy')
    file_paths = glob.glob(search_pattern)
    
    if not file_paths:
        print(f"⚠️ No .npy files found in directory: {data_dir}")
        return None

    all_words = []
    
    print(f"✅ Found {len(file_paths)} files in total.")
    
    # 1. Extract the word from each filename
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        try:
            # Assuming the file format is: WORD_Subject_Session_...
            word = base_name.split('_')[0]
            all_words.append(word)
        except IndexError:
            print(f"Skipping file with unexpected name format: {filename}")
            continue

    # 2. Use Counter to get the unique words and their counts
    word_counts = Counter(all_words)
    
    # 3. Sort the results using the most_common() method
    # This returns a list of (word, count) tuples, sorted by count descending.
    sorted_counts = word_counts.most_common()
    
    return sorted_counts

# --- Execution ---
sorted_word_results = count_and_sort_words(DATA_DIR)

if sorted_word_results:
    # The number of unique words is simply the length of the sorted list
    num_unique_words = len(sorted_word_results)
    
    print("\n--- Summary ---")
    print(f"There are **{num_unique_words}** different (unique) words in the dataset.")
    print("--- Word Counts (Sorted by Occurrences, Highest to Lowest) ---")
    
    # Print the results in a clear list
    for word, count in sorted_word_results:
        if count > 150:
            print(f"- {word}: {count} occurrences")