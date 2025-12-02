import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


from feature_extraction import get_extracted_features

data_file = r"data/train_data.pkl"
labels_file = r"data/train_labels.pkl"

def main():
    data = np.load(data_file, allow_pickle=True)
    labels = np.load(labels_file, allow_pickle=True)
    features, labels = get_extracted_features(data, labels, methods_to_use=[])

    transposed_features = features.T
    print(transposed_features.shape)    # (105, 1466)
    print(len(labels))                  # 1466

    for i in range(len(transposed_features)):
        plt.figure()
        groups = defaultdict(list)
        for val, lab in zip(features[:, i], labels):
            groups[lab].append(val)

        data = [groups[lab] for lab in groups]
        plt.boxplot(data, labels=list(groups.keys()))
        plt.title(f"Feature {i} grouped by label")
    plt.show()


if __name__ == "__main__":
    main()