import os

import libemg
from libemg.emg_predictor import EMGClassifier
import numpy as np
from sklearn.model_selection import KFold

from load_word_files import load_word_files

# might need to remove channel 5

folder = './dataset_words'
target_words = ['THE', 'A', 'TO', 'OF', 'IN', 'ARE', 'AND', 'IS']
# target_words = ['BETTER']

# dict of features in load word files
features_to_use = {'n_zero_crossing':[], 'rms':[], 'n_samples':0}

# What words/classes the model trains with. Good for ignoring final test data.
train_word_dict = load_word_files(folder, target_words, features_to_use, testing_data=True)
test_word_dict = load_word_files(folder, target_words, features_to_use, testing_data=False)

def filter_data(data):
    filter = libemg.filtering.Filter(600)

    if not isinstance(data, np.ndarray):
        data = np.array(data)  # Ensure it's a numpy array

    # data = np.load(r"dataset_words\AFTERWARDS_002_101_0366_3.npy")

    # create a notch filter for removing power line interference
    filter_dictionary = {"name": "notch", "cutoff": 50, "bandwidth": 3}

    filter_dictionary2 = {"name": "highpass", "cutoff": 20, "order": 2}

    filter.install_filters(filter_dictionary=filter_dictionary)

    filter.install_filters(filter_dictionary=filter_dictionary2)

    filtered_data = filter.filter(data)
    return filtered_data

def get_data_labels(word_dict):
    data = []
    labels = []
    for word in word_dict:
        for i, file in enumerate(word_dict[word]):
            data.append([]) # New word
            labels.append(word)
            # print(file[1].values)
            file[1] = filter_data(file[1].values)
            # print(file[1])
            for chl in file[1]:
                signal = chl

                data[-1].append(signal)
            data[-1] = np.array(data[-1])
            # print(data[-1].shape)
            data[-1] = data[-1].transpose()
            # print(data[-1].shape)
    return data, labels

train_data, train_labels = get_data_labels(train_word_dict)
# print(train_data)
# outpath = os.path.join('./data', 'train_data.npy')
# np.save(outpath, train_data)
# outpath = os.path.join('./data', 'train_labels.npy')
# np.save(outpath, train_labels)

test_data, test_labels = get_data_labels(test_word_dict)
# outpath = os.path.join('./data', 'test_data.npy')
# np.save(outpath, test_data)
# outpath = os.path.join('./data', 'test_labels.npy')
# np.save(outpath, test_labels)

# # After loading all of your words for a subject you should have an array of shape (num_words, channels, time)
# data = np.zeros(100, 6, 15000)  # This would mean 100 words, 6 channels, 1500 timepoints

# Assuming your data is in this format it should be correct to extract features from
# You are assuming each word is a single window
fe = libemg.feature_extractor.FeatureExtractor()
# print(fe.get_feature_list())

# # Assuming all your words are the same size you can pass the whole array
# feats = fe.extract_features(['RMS'], data, array=True)

# Assuming your words are of different size you will have to load through them and extract features from each
features = np.array([fe.extract_features(['RMS', 'ZC'], [d], array=True)[0] for d in train_data])
# features = np.array([fe.extract_feature_group('HTD', [d], array=True) for d in train_data])
print(features)
print(features.shape)



