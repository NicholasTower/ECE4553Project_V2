from matplotlib import pyplot as plt

pca_sfs_svm = [32.67, 36.56, 38.4, 38.61, 38.54, 39.22, 39.43, 39.36, 39.29, 39.36, 39.84, 40.72, 40.52, 40.79, 40.65, 41.2, 41.74, 41.95, 42.22, 42.91]
plt.plot(range(1, len(pca_sfs_svm)+1), pca_sfs_svm)
plt.xlabel('Number of components decided by SFS')
plt.ylabel('Accuracy')
plt.title('SVM Classifier With 90% PCA and SFS')
plt.grid(True)
plt.show()