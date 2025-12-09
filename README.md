# Add EMG-UKA-Trial-Corpus to the directory from: https://www.kaggle.com/datasets/xabierdezuazo/emguka-trial-corpus?resource=download
# Run extract_words.py to create a folder with 15757 files. These are all the words extracted from the dataset into numpy files.
# The first 500 instances of the "THE" should be removed manually.
# Running libemg_data_formatting.py will create the approprita .pkl files to classify the training and test words.
# Running kFold_classifier will do all the steps to classify the chosen .pkl files (chosen at the top)
# Run "Final Test.py" will classify the test words after automatically training with the training words.

# You can run "compare_EMG.py" to create a graph with a couple instances of data for a chosen word.
# You can run "compare_two_EMG.py" to create a graph to compare two different words.