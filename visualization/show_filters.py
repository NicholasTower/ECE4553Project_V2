import libemg
import numpy as np

def show_filters():
    filter = libemg.filtering.Filter(600)

    data = np.load(r"dataset_words\AFTERWARDS_002_101_0366_3.npy")

    data = np.delete(data, 4, axis=1)  # remove channel 4

    # create a notch filter for removing power line interference
    filter_dictionary = {"name": "notch", "cutoff": 50, "bandwidth": 3}

    filter_dictionary2 = {"name": "highpass", "cutoff": 20, "order": 2}

    filter.install_filters(filter_dictionary=filter_dictionary)
    filter.install_filters(filter_dictionary=filter_dictionary2)

    filter.visualize_filters()
    filter.visualize_effect(data)    

show_filters()