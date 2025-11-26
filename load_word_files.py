import copy
import glob
import os
import random

import numpy as np
import pandas as pd
# The format of the word dict is:
# [class_words][file][file_name, file_contents, a dict of features (This is empty until word dict is ran through feature_extraction)]

folder = './dataset_words'

# What words/classes the model trains with. Good for ignoring final test data.
target_words = ['A', 'THIS', 'ARE', 'AS']

# dict of features in load word files
features_to_use = {'zero_crossing':[], 'frequency':[]}

def load_word_files(folder, target_words, features_to_use, drop_signal=None, testing_data=True):

    word_file_names = dict()

    for word in target_words:
        word_file_names[word] = []

    for file in glob.glob(os.path.join(folder, '*.csv')):
        # print(file)
        if file[file.find('\\')+1:file.find('_0')] in target_words:
            word_file_names[file[file.find('\\')+1:file.find('_0')]].append([file, [], copy.deepcopy(features_to_use)])

    for word in word_file_names:
        random.seed(395)
        random.shuffle(word_file_names[word])
        if testing_data:
            word_file_names[word] = word_file_names[word][:int(len(word_file_names[word])/2)]
        else:
            word_file_names[word] = word_file_names[word][int(len(word_file_names[word])/2):]

        for i, file in enumerate(word_file_names[word]):
            # print(file[0])
            word_file_names[word][i][1] = pd.read_csv(file[0], names=[0, 1, 2, 3, 4, 5])
            if drop_signal is not None:
                word_file_names[word][i][1].drop(drop_signal)


        # print(word_file_names[word][0][1])
        # print(word_file_names[word][0][1].shape)

    return word_file_names

word_file_names = load_word_files(folder, target_words, features_to_use)