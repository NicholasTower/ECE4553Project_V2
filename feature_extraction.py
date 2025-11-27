import warnings
warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="pygame.pkgdata"
    )
import libemg
import numpy as np

data_file = r"data/train_data.pkl"
labels_file = r"data/train_labels.pkl"

def feature_list_loop(feature_list, data, labels):
    fe = libemg.feature_extractor.FeatureExtractor()
    # Assuming your words are of different size you will have to load through them and extract features from each
    features = np.array([fe.extract_features([feature_list], [d], array=True)[0] for d in data])
    # features = np.array([fe.extract_feature_group('HTD', [d], array=True) for d in train_data])
    return features

def get_extracted_features(data, labels):
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
    # print(features)
    # print(features.shape)

    return features, labels

data = np.load(data_file, allow_pickle=True)
labels = np.load(labels_file, allow_pickle=True)
get_extracted_features(data, labels)
# for feature_list in feature_group_list:
#     print(f'Testing feature set: {feature_list}')
#     feature_list_loop(feature_list, data, labels)