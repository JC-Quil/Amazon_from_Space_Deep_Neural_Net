import numpy as np
import pandas as pd
import csv

from sklearn.metrics import fbeta_score
from keras.models import Model
from keras.callbacks import History
from Keras_Xception_derived_model import Xception_derived_model, data_processing
from utils import WriteDictToCSV



# Open the csv file
df_train = pd.read_csv('data/train_short.csv')

# fix random seed for reproducibility
seed = 11
np.random.seed(seed)
x_train = []
y_train = []
    
# Process data
x_train, y_train = data_processing(df_train, x_train, y_train)

# Define how much pictures to use for train and test sets
###split = 35000 # On full dataset
split1 = 5
###x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:] # On full dataset
x_train, x_valid, y_train, y_valid = x_train[:split1], x_train[split1:], y_train[:split1], y_train[split1:]
print(x_train.shape)
print(y_train.shape)

# Model definition
model = Xception_derived_model(include_top=False, weights=None)
# Compile model
model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
# Fit model
history = History()
model.fit(x_train, y_train,
      batch_size=1,
      epochs=1,
      verbose=0,
      validation_data=(x_valid, y_valid),
      callbacks=[history])

# list all data in history
print(history.history)
print history.history.keys()
WriteDictToCSV('history.csv', history.history.keys(), history.history)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

p_valid = model.predict(x_valid, batch_size=1)
print(y_valid)
print(p_valid)
eval_score = [fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')]
print(eval_score[0])
np.savetxt("fbeta_score.csv", eval_score, delimiter=",")
