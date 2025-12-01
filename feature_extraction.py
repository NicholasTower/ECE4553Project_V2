import libemg
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

data = np.load(r"data/test_data.pkl", allow_pickle=True)
labels = np.load(r"data/test_labels.pkl", allow_pickle=True)

# # After loading all of your words for a subject you should have an array of shape (num_words, channels, time)
# data = np.zeros(100, 6, 15000)  # This would mean 100 words, 6 channels, 1500 timepoints

# Assuming your data is in this format it should be correct to extract features from
# You are assuming each word is a single window
fe = libemg.feature_extractor.FeatureExtractor()
# print(fe.get_feature_list())

HTD = ['MAV','ZC','WL']
# TDAR = ['MAV','ZC','SSC','WL','AR4'] SCC and AR4 require windows
LS4 = ['WAMP'] # Removed LS, MFL and MSR as they require windows
# TDPSD = ['M0','M2','M4','SPARSI','IRF','WLF'] none worked
TSTD = ['DASDV','WAMP','ZC','MFL','SAMPEN',] #removed *TDPSD, MAVFD, DASDV, MFL, SAMPEN
Combined = ['WL','SCC','LD','AR9'] # 
# feature_group_list = [HTD, TDAR, LS4, TDPSD, TSTD, Combined]

features = np.array([fe.extract_features('ZC', [d], array=True) for d in data])
print(features)
print(features.shape)


def feature_list_loop(feature_list, data, labels):
    # Assuming your words are of different size you will have to load through them and extract features from each
    features = np.array([fe.extract_features([feature_list], [d], array=True)[0] for d in data])
    # features = np.array([fe.extract_feature_group('HTD', [d], array=True) for d in train_data])

    def test_features(features, labels, classifier='LDA'):
        percentages = []
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

    test_features(features, labels, classifier='LDA')
# test_features(features, labels, classifier='SVM')
# test_features(features, labels, classifier='KNN')
# test_features(features, labels, classifier='QDA')

# for feature_list in feature_group_list:
#     print(f'Testing feature set: {feature_list}')
#     feature_list_loop(feature_list, data, labels)