import copy
import glob
import os

import numpy as np
import pandas as pd

folder = './dataset_words'
target_words = ['A', 'THIS', 'ARE', 'AS']

# dict of features in load word files
features_to_use = {'zero_crossing':[], 'frequency':[]}

def load_word_files(folder, target_words, features_to_use, drop_signal=None):

    word_file_names = dict()

    for word in target_words:
        word_file_names[word] = []

    for file in glob.glob(os.path.join(folder, '*.csv')):
        # print(file)
        if file[file.find('\\')+1:file.find('_0')] in target_words:
            word_file_names[file[file.find('\\')+1:file.find('_0')]].append([file, [], copy.deepcopy(features_to_use)])

    for word in word_file_names:
        for i, file in enumerate(word_file_names[word]):
            # print(file[0])
            word_file_names[word][i][1] = pd.read_csv(file[0], names=[0, 1, 2, 3, 4, 5])
            if drop_signal is not None:
                word_file_names[word][i][1].drop(drop_signal)
        # print(word_file_names[word][0][1])
        # print(word_file_names[word][0][1].shape)

    return word_file_names

word_file_names = load_word_files(folder, target_words, features_to_use)