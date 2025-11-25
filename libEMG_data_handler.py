from libemg.data_handler import OfflineDataHandler, RegexFilter

# Step 1: Initialize the OfflineDataHandler
odh = OfflineDataHandler()

# Step 2: Load data from a specified folder using regex filters for metadata
dataset_folder = 'example_words'
regex_filters = [
    RegexFilter(right_bound="_", description='classes')
]
odh.get_data(folder_location=dataset_folder, regex_filters=regex_filters, delimiter=",")