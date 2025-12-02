# This file can be run to take the .npy single word files made by extract_words.py and create pickle filed of the data
# and labels for testing & training, filtered & unfiltered, and channel 5 (1-6) dropped & not dropped.
#
#
#
#
#
#

import os

import warnings         #Gets rid of pygame warning
warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="pygame.pkgdata"
    )
import libemg
from libemg.emg_predictor import EMGClassifier
import numpy as np
import pickle

from load_word_files import load_word_files

gen_pkl = True
# gen_pkl = False

# might need to remove channel 5

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

def get_data_labels(word_dict, filter=True):
    data = []
    labels = []
    for word in word_dict:
        for i, file in enumerate(word_dict[word]):
            data.append([]) # New word
            labels.append(word)
            # print(file[1].values)
            if filter:
                file[1] = filter_data(file[1])
            # print(file[1])
            for chl in file[1]:
                signal = chl

                data[-1].append(signal)
            data[-1] = np.array(data[-1])
            # print(data[-1].shape)
            data[-1] = data[-1].transpose()
            # print(data[-1].shape)
    return data, labels

def generate_pickles(filter=True, drop_signal=None):
    folder = './dataset_words'
    target_words = ['THE', 'A', 'TO', 'OF', 'IN', 'ARE', 'AND', 'IS']
    # target_words = ['BETTER']

    # dict of features in load word files
    features_to_use = {'n_zero_crossing': [], 'rms': [], 'n_samples': 0}

    print(f'Generating train pickle files... {'Filtered' if filter else 'Not Filtered'} '
          f'{'Not Dropped' if drop_signal is None else 'Dropped'}')
    train_word_dict = load_word_files(folder, target_words, features_to_use, testing_data=True, drop_signal=drop_signal)
    train_data, train_labels = get_data_labels(train_word_dict, filter=filter)
    outpath = os.path.join('./data', f'train_data{'_nfiltered' if not filter else ''}'
                                     f'{'_ndrop' if drop_signal is None else ''}.pkl')
    with open(outpath, 'wb') as file:
        pickle.dump(train_data, file)
    # np.save(outpath, train_data)
    if filter and drop_signal is None:
        outpath = os.path.join('./data', 'train_labels.pkl')
        with open(outpath, 'wb') as file:
            pickle.dump(train_labels, file)
    # np.save(outpath, train_labels)

    print(f'Generating test pickle files... {'Filtered' if filter else 'Not Filtered'} '
          f'{'Not Dropped' if drop_signal is None else 'Dropped'}')
    test_word_dict = load_word_files(folder, target_words, features_to_use, testing_data=False, drop_signal=drop_signal)
    test_data, test_labels = get_data_labels(test_word_dict, filter=filter)
    outpath = os.path.join('./data', f'test_data{'_nfiltered' if not filter else ''}'
                                     f'{'_ndrop' if drop_signal is None else ''}.pkl')
    with open(outpath, 'wb') as file:
        pickle.dump(test_data, file)
    # np.save(outpath, test_data)
    if filter and drop_signal is None:
        outpath = os.path.join('./data', 'test_labels.pkl')
        with open(outpath, 'wb') as file:
            pickle.dump(test_labels, file)
    # np.save(outpath, test_labels)

if gen_pkl:
    generate_pickles()
    generate_pickles(drop_signal=4)
    generate_pickles(filter=False)
    generate_pickles(filter=False, drop_signal=4)

print('loading train set')
data = pickle.load(open('./data\\train_data.pkl', 'rb'))
print(len(data), len(data[0]), len(data[0][0]))
labels = pickle.load(open('./data\\train_labels.pkl', 'rb'))
# print(labels)

# # After loading all of your words for a subject you should have an array of shape (num_words, channels, time)
# data = np.zeros(100, 6, 15000)  # This would mean 100 words, 6 channels, 1500 timepoints

# # Assuming your data is in this format it should be correct to extract features from
# # You are assuming each word is a single window
# fe = libemg.feature_extractor.FeatureExtractor()
# # print(fe.get_feature_list())
#
# # Assuming your words are of different size you will have to load through them and extract features from each
# features = np.array([fe.extract_features(['RMS', 'ZC'], [d], array=True)[0] for d in data])
# # features = np.array([fe.extract_feature_group('HTD', [d], array=True) for d in train_data])
# print(features)
# print(features.shape)
#
# clf = EMGClassifier('LDA')
# clf.fit(features, labels)


