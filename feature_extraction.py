# This file is where the features from of the word recordings are extracted using libemg and the reduction is applied
# using various methods.
import warnings
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SequentialFeatureSelector

warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="pygame.pkgdata"
    )
import libemg
import numpy as np
from sklearn.preprocessing import StandardScaler

data_file = r"data/train_data.pkl"
labels_file = r"data/train_labels.pkl"

def feature_list_loop(feature_list, data, labels):
    fe = libemg.feature_extractor.FeatureExtractor()
    # Assuming your words are of different size you will have to load through them and extract features from each
    features = np.array([fe.extract_features([feature_list], [d], array=True)[0] for d in data])
    # features = np.array([fe.extract_feature_group('HTD', [d], array=True) for d in train_data])
    return features

def plotLDA(labels, features, dataset, plot=True):
    print('Performing LDA...')
    lda = LinearDiscriminantAnalysis()
    scores = lda.fit_transform(features, labels)

    # if (plot):
    #     cmap = plt.get_cmap('tab10')
    #     colors = {lab: cmap(i%10) for i, lab in enumerate(np.unique(labels))}

    #     plt.xlabel("LD1")
    #     plt.title("LDA Projection for "+dataset)
    #     plt.grid(True)

    #     for lab in np.unique(labels):
    #         mask = (labels == lab)
    #         if (scores.shape[1] == 1):
    #             plt.scatter(scores[mask, 0], np.random.normal(loc=0, scale = 0.05, size=sum(mask)), label=str(lab), color=colors[lab], s=50)
    #             plt.yticks([])  # remove y-axis, it's meaningless
    #         else:
    #             plt.scatter(scores[mask, 0], scores[mask, 1], label=str(lab), color=colors[lab], s=50, alpha=0.3, edgecolors="black")
    #             plt.ylabel("LD2")

    #     plt.legend()
    #     plt.show()

    return scores

def apply_pca(old_features, old_labels, variance_kept=0.90, show_plots=False):
    print('Performing PCA...')
    # Principle Components:
    pca = PCA(n_components=variance_kept, random_state=0)
    transformed_features = pca.fit_transform(old_features)
    eigenvectors = pca.explained_variance_ratio_
    cumsum = 0
    needed = []
    # print(len(eigenvectors))
    for i in eigenvectors:
        cumsum += i*100
        needed.append(cumsum/100)

    plt.figure()
    plt.plot(range(1, len(eigenvectors) + 1), eigenvectors, color="red", marker="o")
    plt.plot(range(1, len(eigenvectors) + 1), needed, color="blue", marker="o")
    plt.bar(range(1, len(eigenvectors) + 1), eigenvectors)
    plt.xlabel("Eigenvector")
    plt.ylabel("Eigenvalue")
    plt.title(f"Scree Plot of Dataset Words with PCA")
    plt.grid(True)
    # plt.show()

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    # print(loadings)
    plt.figure()
    if len(transformed_features[0]) > 1:
        plt.scatter(transformed_features[:, 0], transformed_features[:, 1], alpha=0.7, label='Observations', s=5)
        u = loadings[:, 0]
        v = loadings[:, 1]

        angles = np.arctan2(v, u)
        norm = Normalize()
        norm.autoscale(angles)
        colormap = cm.inferno
        # norm = plt.Normalize(angles.min(), angles.max())
        # cmap = cm.hsv
        plt.quiver([0] * len(u), [0] * len(v), u, v, color=colormap(norm(angles)), alpha=0.8, scale=3,
                   angles='xy')
        # for i in range(len(u)):
        #     plt.text(u[i], v[i], columns)

        plt.xlabel(f'Principal Component 1 (Explained Variance: {eigenvectors[0]:.2f})')
        plt.ylabel(f'Principal Component 2 (Explained Variance: {eigenvectors[1]:.2f})')

    plt.title(f'PCA Biplot')
    plt.grid(True)
    plt.axhline(0, color='grey', linewidth=0.5)
    plt.axvline(0, color='grey', linewidth=0.5)
    if show_plots:
        plt.show()

    # ICA
    # ica = FastICA(n_components=1, random_state=0)
    # vectors = ica.fit_transform(scaled_features)

    print(f"Variance kept of Training Dataset: {cumsum:.2f}% using {len(needed)} component(s)\n")
    return transformed_features, old_labels     # PCA
    # return vectors, old_labels                  # ICA

def sfs_optimizer(features, labels, n):
    print('Performing SFS...')
    clf = LinearDiscriminantAnalysis()
    sfs = SequentialFeatureSelector(clf, n_features_to_select=n)
    sfs.fit(features, labels)
    good_feature_indexes = sfs.get_support(indices=True)
    print(f"Optimal features to use n={n}: {good_feature_indexes}")

    optimized_features = []
    for j in range(len(features)):
        optimized_features.append([])
        for i in good_feature_indexes:
            optimized_features[j].append(float(features[j][i]))
    return optimized_features

def get_extracted_features(data, labels, variance_kept=0.9, show_plots=False, methods_to_use=['pca'], n='auto'):
    # # After loading all of your words for a subject you should have an array of shape (num_words, channels, time)
    # data = np.zeros(100, 6, 15000)  # This would mean 100 words, 6 channels, 1500 timepoints

    # Assuming your data is in this format it should be correct to extract features from
    # You are assuming each word is a single window
    fe = libemg.feature_extractor.FeatureExtractor()
    # print(fe.get_feature_list())

    HTD = ['MAV','ZC','WL']
    TDAR = ['MAV','ZC','WL','AR4']
    LS4 = ['LS','MFL','MSR','WAMP']
    TDPSD = ['M0','M2','M4','SPARSI','IRF','WLF']
    TSTD = ['MAVFD','DASDV','WAMP','ZC','MFL','SAMPEN',*TDPSD]
    Combined = ['WL','SCC','LD','AR9']
    # print(TSTD)
    feature_group_list = [HTD, TDAR, LS4, TDPSD, TSTD, Combined]

    # List from
    possible_features = ['MAV', 'ZC', 'WL', 'MFL', 'WAMP', 'RMS', 'IAV', 'DASDV', 'VAR', 'LD', 'MAVFD', 'SKEW', 'KURT',
                         'WENG']
    gives_warnings = ['WV', 'WWL', 'WENT', 'MEAN']

    # # Tests all features
    # for feat in possible_features:
    #     print(feat)
    #     features = np.array([fe.extract_features([feat], [d], array=True)[0] for d in data])

    # features = np.array([fe.extract_features(['MAV'], [d], array=True)[0] for d in data])
    features = np.array([fe.extract_features(possible_features, [d], array=True)[0] for d in data])
    ss = StandardScaler()
    features = ss.fit_transform(features)
    # print(features)
    # print(features.shape)

    for method in methods_to_use:
        if method == 'pca':
            features, labels = apply_pca(features, labels, variance_kept=variance_kept, show_plots=show_plots)
        if method == 'lda':
            features = plotLDA(labels, features, "Training Dataset", plot=show_plots)
        if method == 'sfs':
            features = sfs_optimizer(features, labels, n=n)


    return features, labels

def main():
    data = np.load(data_file, allow_pickle=True)
    labels = np.load(labels_file, allow_pickle=True)
    get_extracted_features(data, labels, variance_kept=0.90, show_plots=True)
    # for feature_list in feature_group_list:
    #     print(f'Testing feature set: {feature_list}')
    #     feature_list_loop(feature_list, data, labels)

if __name__ == "__main__":
    main()
