import numpy as np
import pandas as pd
import csv

from sklearn.metrics import fbeta_score
from keras.models import Model
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.callbacks import History
from Keras_Xception_derived_gen import Xception_derived_model, data_processing 
from utils import batch_generator
from utils import WriteDictToCSV, create_folders


# Open the csv file
#df_train = pd.read_csv('data/train_short.csv')
#print "df_train", df_train.shape
#df_valid = pd.read_csv('data/valid_short.csv')
#print "df_valid", df_valid.shape
#df_train = pd.read_csv('data/train_med.csv')
#print "df_train", df_train.shape
#df_valid = pd.read_csv('data/valid_med.csv')
#print "df_valid", df_valid.shape
df_train = pd.read_csv('data/train_v2.csv')
print "df_train", df_train.shape
df_valid = pd.read_csv('data/valid_v2.csv')
print "df_valid", df_valid.shape
df_main = pd.read_csv('data/main_v2.csv')

# Model definition
model = Xception_derived_model(include_top=False, weights=None)
epochs = 120
learning_rate = 0.06
decay_rate = learning_rate / epochs
momentum = 0.9
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

# Compile model
model.compile(loss= 'binary_crossentropy' , optimizer= sgd , metrics=[ 'accuracy' ])
print("model compiled")

history = History()

# Define the number of steps for all the batches
steps_per_epoch = 4048
validation_steps = 1010

# Fit the model using the generator a batch at a time (no batch_size option)
model.fit_generator(generator = batch_generator(df_train, "train"), steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=0, callbacks=[history], validation_data = batch_generator(df_valid, "valid"), validation_steps = validation_steps)


# list and save all data in history
print(history.history)
WriteDictToCSV('history.csv', history.history.keys(), history.history)

# Save the model
model.save("complete_model.h5")
#serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
