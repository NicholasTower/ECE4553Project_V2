import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from manual_feature_extraction import feature_extraction

folder = './dataset_words'
target_words = ['A', 'THIS', 'ARE', 'AS']
# words = ['BETTER']

# dict of features in load word files. These are the names of the signals
features_to_use = {'n_zero_crossing':[], 'rms':[], 'n_samples':0}

def sort_features(words_dict):
    labels = []
    features = []
    for word in words_dict:
        for file in words_dict[word]:
            labels.append(word)
            features.append([])
            for feature in file[2]:
                try:
                    for f in file[2][feature]:
                        # print(f)
                        features[len(features) - 1].append(f)
                except TypeError:
                    features[len(features)-1].append(file[2][feature])

    return features, labels

def test_features(folder, target_words, features_to_use, drop_signal=None, features=None, labels=None, classifier='LDA'):
    percentages = []
    if features is None:
        features, labels = sort_features(feature_extraction(folder, target_words, features_to_use, drop_signal=drop_signal))
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(features)

    print(f'Beginning KFold with {classifier} splits')
    for i, (train_index, test_index) in enumerate(kf.split(features)):
        # print(f"Fold {i}:")
        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")
        if classifier == 'LDA':
            clf = LinearDiscriminantAnalysis()
        elif classifier == 'SVM':
            clf = SVC()
        elif classifier == 'KNN':
            clf = KNeighborsClassifier()
        elif classifier == 'QDA':
            clf = QuadraticDiscriminantAnalysis()
        train_features = []
        train_labels = []
        for j in train_index:
            train_features.append(features[j])
            train_labels.append(labels[j])

        # print('Train features', train_features)
        # print('Train labels', train_labels)
        clf.fit(train_features, train_labels)

        test_features = []
        test_labels = []
        for j in test_index:
            test_features.append(features[j])
            test_labels.append(labels[j])

        # print(f"   Real: index={test_labels}")
        resultant_prediction = clf.predict(test_features)
        # print(f"Guesses: {resultant_prediction}")
        total_correct = 0
        for k in range(len(test_labels)):
            if test_labels[k] == resultant_prediction[k]:
                total_correct += 1
        percentages.append((total_correct / len(test_labels)) * 100)
        print(f"   Fold: {i}  Correct: {total_correct} -> {round(percentages[len(percentages) - 1], 2)}%")
    average_percentage = np.mean(percentages)
    standard_deviation = np.std(percentages)
    print(f"Average: {round(average_percentage, 2)}%")
    print(f"     SD: {round(standard_deviation, 2)}%\n\n")
    all_percentages = dict()
    all_percentages["Standard Deviation"] = standard_deviation
    all_percentages["All"] = percentages
    all_percentages["Average"] = average_percentage
    return all_percentages

features, labels = sort_features(feature_extraction(folder, target_words, features_to_use))
print(features)
print(features)
test_features(folder, target_words, features_to_use, features=features, labels=labels)
test_features(folder, target_words, features_to_use, features=features, labels=labels, classifier='SVM')
test_features(folder, target_words, features_to_use, features=features, labels=labels, classifier='KNN')
test_features(folder, target_words, features_to_use, features=features, labels=labels, classifier='QDA')