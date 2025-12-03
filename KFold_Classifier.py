# This file will run a kfold on different classifiers. It uses det_extracted_features() from feature_extraction.py to
# get the features and labels it uses. The labels are passed through in case the feature extraction stuff changes things.
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from feature_extraction import get_extracted_features

data_file = r"data\train_data_ndrop_fewer_the.pkl"
labels_file = r"data\train_labels_fewer_the.pkl"

def kfold_classifier(features, labels, classifier='LDA', svm_kernal='rbf', plot_confusion_matrix=False):

    class_names = np.unique(labels)
    all_true_labels = []
    all_predictions = []

    percentages = []
    kf = KFold(n_splits=8, shuffle=True, random_state=0)
    kf.get_n_splits(features)

    print(f'Beginning KFold with {classifier} {svm_kernal if classifier=="SVM" else ''} splits')
    for i, (train_index, test_index) in enumerate(kf.split(features)):
        # print(f"Fold {i}:")
        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")
        if classifier == 'LDA':
            clf = LinearDiscriminantAnalysis()
        elif classifier == 'SVM':
            clf = SVC(kernel=svm_kernal)
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

        all_true_labels.extend(test_labels)
        all_predictions.extend(resultant_prediction)

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

    if plot_confusion_matrix:
        plot_confusion_matrix(all_true_labels, all_predictions, class_names, classifier)

    all_percentages = dict()
    all_percentages["Standard Deviation"] = standard_deviation
    all_percentages["All"] = percentages
    all_percentages["Average"] = average_percentage
    return all_percentages

def plot_confusion_matrix(true_labels, pred_labels, class_names, classifier_name):
    """Calculates and plots the confusion matrix across all test folds using string labels."""
    
    # Calculate the Confusion Matrix
    # We explicitly pass the class_names to ensure the matrix axes are correctly ordered.
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names) 
    
    # Display the Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical', values_format='d')
    
    ax.set_title(f'Confusion Matrix for {classifier_name} (Aggregated K-Fold) without Reduced "the" Class')
    plt.tight_layout()
    plt.show()

    # Optional: Print Class-Wise Recall
    print("\n--- Class-Wise Recall (Accuracy per Class) ---")
    for i, name in enumerate(class_names):
        true_positives = cm[i, i]
        total_actual = np.sum(cm[i, :])
        recall = (true_positives / total_actual) if total_actual > 0 else 0
        print(f"  {name}: {true_positives} / {total_actual} Correct ({recall:.2f})")
    print("------------------------------------------\n")

def main():
    print("Loading data & labels...")
    data = np.load(data_file, allow_pickle=True)
    labels = np.load(labels_file, allow_pickle=True)

    print("Getting features...")
    # for i in range(1, 7):
    # features, labels = get_extracted_features(data, labels, variance_kept=0.9, methods_to_use=['pca', 'sfs'], n=i)

    # features, labels = get_extracted_features(data, labels, variance_kept=0.9, methods_to_use=['pca'])
    # features, labels = get_extracted_features(data, labels, variance_kept=0.9, methods_to_use=['sfs', 'lda'])
    features, labels = get_extracted_features(data, labels, variance_kept=0.9, methods_to_use=['lda'])
    kfold_classifier(features, labels, classifier='LDA')
    # kfold_classifier(features, labels, classifier='QDA')
    # kfold_classifier(features, labels, classifier='KNN')
    # kfold_classifier(features, labels, classifier='SVM', svm_kernal='rbf')
    # kfold_classifier(features, labels, classifier='SVM', svm_kernal='linear')
    # kfold_classifier(features, labels, classifier='SVM', svm_kernal='sigmoid')
    # kfold_classifier(features, labels, classifier='SVM', svm_kernal='poly')

if __name__ == "__main__":
    main()