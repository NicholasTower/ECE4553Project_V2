import numpy as np
import libemg

filter = libemg.filtering.Filter(600)

data = np.load(r"dataset_words\AFTERWARDS_002_101_0366_3.npy")

# create a notch filter for removing power line interference
filter_dictionary={ "name": "notch", "cutoff": 50, "bandwidth": 3}
filter_dictionary2={ "name": "highpass", "cutoff": 20, "order":2}
filter.install_filters(filter_dictionary=filter_dictionary)
filter.install_filters(filter_dictionary=filter_dictionary2)
filtered_data = filter.filter(data)

filtered_data = filtered_data.transpose()

fe = libemg.feature_extractor.FeatureExtractor()

feats = np.array([fe.extract_features(['RMS'], np.array([filtered_data]), array=True)][0])

print("feature:")
print(feats)