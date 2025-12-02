import libemg
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


def plotLDA(train_features, test_features, labels):
    print('Performing LDA...')
    lda = LinearDiscriminantAnalysis()
    train = lda.fit_transform(train_features, labels)
    test = lda.transform(test_features)
    # print(scores)

    return train, test

def extract_features(train_data, test_data, labels):
    fe = libemg.feature_extractor.FeatureExtractor()
    possible_features = ['MAV', 'ZC', 'WL', 'MFL', 'WAMP', 'RMS', 'IAV', 'DASDV', 'VAR', 'LD', 'MAVFD', 'SKEW', 'KURT',
                         'WENG']

    train_features = np.array([fe.extract_features(possible_features, [d], array=True)[0] for d in train_data])
    ss = StandardScaler()
    train_features = ss.fit_transform(train_features)

    test_features = np.array([fe.extract_features(possible_features, [d], array=True)[0] for d in test_data])
    test_features = ss.transform(test_features)

    train_features, test_features = plotLDA(train_features, test_features, labels)
    return train_features, test_features

train_data_file = r"data\train_data_ndrop_fewer_the.pkl"
test_data_file = r"data\test_data_ndrop_fewer_the.pkl"
labels_file = r"data\train_labels_fewer_the.pkl"

train_data = np.load(train_data_file, allow_pickle=True)
test_data = np.load(test_data_file, allow_pickle=True)
labels = np.load(labels_file, allow_pickle=True)


train_features, test_features = extract_features(train_data, test_data, labels)

