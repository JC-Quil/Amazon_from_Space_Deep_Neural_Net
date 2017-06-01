import numpy as np
import pandas as pd
import csv
import os


def WriteDictToCSV(csv_file,csv_columns,dict_data):
    """
    Copy the model training history file as a csv file.
    """
    with open('history.csv','wb') as csvfile:
        w = csv.writer(csvfile)
        w.writerows(dict_data.items())

    return

def create_folders():
    """
    Create the necessary folders
    """
    if not os.path.exists("data/train-npy/"):
        os.makedirs("data/train-npy/")
    if not os.path.exists("data/test-npy/"):
        os.makedirs("data/test-npy/")
    if not os.path.exists("data/valid-npy/"):
        os.makedirs("data/valid-npy/")

def label_map_gen(df_main):
    """
    Generate the dictionary containing the different labels and their index.
    """
    # Function to flatten a list of list
    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in df_main['tags'].values])))

    # Create list of labels
    label_map = {l: i for i, l in enumerate(labels)}
    return label_map

def batch_generator(labels_df, set_kind):
    """
    Generate batches of images/features for training, validation and testing purpose.
    The batches are kept small due to memory limitations of the GPU.
    """
    # Generate training batches
    if set_kind == "train" and (labels_df.shape[0] == 32384 or labels_df.shape[0] == 3120 or labels_df.shape[0] == 64):
        while 1:

            for i in range(labels_df.shape[0]//8):
                x_train = np.load('data/train-npy/npdatasetX{}.npy'.format(i))
                y_train = np.load('data/train-npy/npdatasetY{}.npy'.format(i))

                for j in range(1):
                    x_trainj = x_train[j*8:j*8-1,:]
                    y_trainj = y_train[j*8:j*8-1,:]

                    yield (x_trainj, y_trainj)


    # Generate validation batches
    if set_kind == "valid" and (labels_df.shape[0] == 8080 or labels_df.shape[0] == 1920 or labels_df.shape[0] == 8):
        while 1:

            for i in range(labels_df.shape[0]//4): 
                x_valid = np.load('data/valid-npy/npdatasetX{}.npy'.format(i))
                y_valid = np.load('data/valid-npy/npdatasetY{}.npy'.format(i))

                for j in range(1): 
                    x_validj = x_valid[j*4:j*4-1,:]
                    y_validj = y_valid[j*4:j*4-1,:]

                    yield (x_validj, y_validj)


    # Generate test batches
    if set_kind == "test" and labels_df.shape[0] == 40669:
        while 1:

            for i in range(labels_df.shape[0]//4): #REPLACE 1 by 3
                x_valid = np.load('data/valid-npy/npdatasetX{}.npy'.format(i))

                for j in range(1): #REPLACE 2 by 2816
                    x_validj = x_valid[j*4:j*4-1,:]
                    
                yield (x_validj, y_validj)

    if set_kind == "test" and (labels_df.shape[0] == 8080 or labels_df.shape[0] == 8):
        while 1:

            for i in range(labels_df.shape[0]//8): #REPLACE 1 by 3
                x_valid = np.load('data/valid-npy/npdatasetX{}.npy'.format(i))

                for j in range(2): #REPLACE 2 by 2816
                    x_validj = x_valid[j*4:j*4-1,:]

                    yield x_validj
