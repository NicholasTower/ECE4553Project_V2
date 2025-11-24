import numpy as np

# 1. Point to your specific file
filename = r"C:\Users\ntowe\Documents\School\ECE4553Project\ECE4553Project_V2\EMG-UKA-Trial-Corpus\emg\008\003\e07_008_003_0215.adc" 

# 2. Read the raw binary data
# "dtype=np.int16" tells Python that every 2 bytes = 1 number
raw_data = np.fromfile(filename, dtype=np.int16)

# 3. Reshape the data
# The file name "e07" and the PDF documentation confirm there are 7 CHANNELS.
# -1 tells numpy to automatically calculate the number of rows (time samples).
emg_signal = raw_data.reshape(-1, 7)

print(f"Data Opened!")
print(f"Shape: {emg_signal.shape} (Time Samples, Channels)")
print(f"First readings of Channel 0:\n{emg_signal[0:39, [6]]}")