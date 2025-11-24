import numpy as np
# Import the function directly, alongside the Handler class
from libemg.data_handler import OfflineDataHandler
from libemg.utils.regex_utils import make_regex

# # Try this later:

# import numpy as np
# from libemg.data_handler import OfflineDataHandler
# import os

# # Assume your data is organized like this:
# # root_folder/
# #   class_1_rep_1.npy
# #   class_1_rep_2.npy
# #   class_2_rep_1.npy
# #   ...

# def load_numpy_data_into_libemg(root_folder):
#     # This dictionary will hold the data in the format libemg expects
#     # The structure should be { 'class_label': [np.array(emg_data), np.array(emg_data), ...] }
#     data_dict = {}

#     for filename in os.listdir(root_folder):
#         if filename.endswith(".npy"):
#             file_path = os.path.join(root_folder, filename)
            
#             # Load the numpy array
#             emg_data = np.load(file_path)
            
#             # Extract the class label from the filename (adjust this based on your naming convention)
#             # Example: from "class_1_rep_1.npy", extract "class_1" or "1"
#             # A more robust method with regex would be better for complex naming schemes
#             class_label = filename.split('_')[0] # Simple example: assumes class is the first part

#             if class_label not in data_dict:
#                 data_dict[class_label] = []
                
#             data_dict[class_label].append(emg_data)

#     # Initialize the OfflineDataHandler with the prepared dictionary
#     # The constructor automatically handles the internal structure from the dict
#     odh = OfflineDataHandler(data_dict)
#     return odh

# if __name__ == "__main__":
#     # Specify the directory where your .npy files are located
#     my_data_folder = 'path/to/your/numpy_data'
    
#     # Load the data
#     if os.path.exists(my_data_folder):
#         handler = load_numpy_data_into_libemg(my_data_folder)
#         print("Data loaded into OfflineDataHandler from numpy files.")
#         # You can now proceed with feature extraction, classification, etc.
#     else:
#         print(f"Error: Folder not found at {my_data_folder}")

