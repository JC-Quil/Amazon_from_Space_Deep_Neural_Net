import numpy as np
import pandas as pd
import csv

from Keras_Xception_derived_gen import Xception_derived_model, data_processing 
from utils import batch_generator
from utils import WriteDictToCSV, create_folders, label_map_gen


# Open the train and validation csv file
#df_train = pd.read_csv('data/train_med.csv')
#print "df_train", df_train.shape
#df_valid = pd.read_csv('data/valid_med.csv')
#print "df_valid", df_valid.shape
#df_train = pd.read_csv('data/train_v2.csv')
#print "df_train", df_train.shape
df_valid = pd.read_csv('data/valid_v2.csv')
print "df_valid", df_valid.shape
df_main = pd.read_csv('data/main_v2.csv')

# Generate the dictionary mapping the labels
label_map = label_map_gen(df_main)
print label_map

# Create necessary folders
create_folders()

#Initiate arrays
x_train = []
y_train = []
x_valid = []
y_valid = []
    
# Process data
data_processing(df_train, x_train, y_train, label_map)
print "test data processed ok"
data_processing(df_valid, x_valid, y_valid, label_map)
print "valid data processed ok"