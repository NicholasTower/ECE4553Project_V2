import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import libemg
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(true_labels, pred_labels, class_names, classifier_name):
    
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names) 
    
    # Display the Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical', values_format='d')
    
    ax.set_title(f'Confusion Matrix using {classifier_name} for Test Set')
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

def plotLDA(train_features, train_labels, test_features):
    print('Performing LDA...')
    lda = LinearDiscriminantAnalysis()
    train = lda.fit_transform(train_features, train_labels)
    test = lda.transform(test_features)
    # print(scores)

    return train, test

def extract_features(train_data, train_labels, test_data):
    fe = libemg.feature_extractor.FeatureExtractor()
    possible_features = ['MAV', 'ZC', 'WL', 'MFL', 'WAMP', 'RMS', 'IAV', 'DASDV', 'VAR', 'LD', 'MAVFD', 'SKEW', 'KURT',
                         'WENG']

    train_features = np.array([fe.extract_features(possible_features, [d], array=True)[0] for d in train_data])
    ss = StandardScaler()
    train_features = ss.fit_transform(train_features)

    test_features = np.array([fe.extract_features(possible_features, [d], array=True)[0] for d in test_data])
    test_features = ss.transform(test_features)

    train_features, test_features = plotLDA(train_features, train_labels, test_features)
    return train_features, test_features

train_data_file = r"data\train_data_ndrop_fewer_the.pkl"
test_data_file = r"data\test_data_ndrop_fewer_the.pkl"
train_labels_file = r"data\train_labels_fewer_the.pkl"
test_labels_file = r"data\test_labels_fewer_the.pkl"

print('Loading data...')
train_data = np.load(train_data_file, allow_pickle=True)
test_data = np.load(test_data_file, allow_pickle=True)
train_labels = np.load(train_labels_file, allow_pickle=True)
test_labels = np.load(test_labels_file, allow_pickle=True)

print('Extracting features...')
train_features, test_features = extract_features(train_data, train_labels, test_data)

clf = LinearDiscriminantAnalysis()
clf.fit(train_features, train_labels)

resultant_prediction = clf.predict(test_features)

total_correct = 0
for k in range(len(test_labels)):
    if test_labels[k] == resultant_prediction[k]:
        total_correct += 1
percentage = (total_correct / len(test_labels)) * 100
print(f"Correct: {total_correct} of {len(test_labels)} -> {round(percentage, 2)}%")

plot_confusion_matrix(test_labels, resultant_prediction, np.unique(test_labels), "LDA")