import numpy as np
import pandas as pd

from load_word_files import load_word_files

folder = './dataset_words'
target_words = ['A', 'THIS', 'ARE', 'AS']
# words = ['BETTER']

# dict of features in load word files. These are the names of the signals
features_to_use = {'n_zero_crossing':[], 'rms':[], 'n_samples':0}

def get_n_zero_crossings(signal):
    crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(crossings)

def get_rms(signal):
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)  # Ensure it's a numpy array
    # print('rms: ', np.sqrt(np.mean(signal**2)))
    return np.sqrt(np.mean(signal**2))

def feature_extraction(folder, target_words, features_to_use, drop_signal=None):
    words_dict = load_word_files(folder, target_words, features_to_use, drop_signal=drop_signal)
    for word in words_dict:
        for i, file in enumerate(words_dict[word]):
            file[2]['n_samples'] = len(file[1][0].values) # gets length of the signals
            for col in file[1]:
                signal = file[1][col].values
                ###############################################
                # Where to put the feature extraction
                n_zero_crossings = get_n_zero_crossings(signal)
                file[2]['n_zero_crossing'].append(n_zero_crossings)

                rms = get_rms(signal)
                file[2]['rms'].append(rms)
                ###############################################

                # print(col)
                # print(file[1][col].values)
                # print(n_zero_crossings)
    # print(words_dict['THIS'][0][2]['n_zero_crossing'])
    # print(len(words_dict['THIS'][0][2]['n_zero_crossing']))
    # print(len(words_dict['THIS'][1][2]['n_zero_crossing']))
    return words_dict

feature_extraction(folder, target_words, features_to_use)
