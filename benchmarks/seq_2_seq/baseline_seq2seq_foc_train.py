#!/usr/bin/env/python3
# Author: David Jonietz
# Copyright 2019 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
# IARAI licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Simple seq2seq baseline in keras with convLSTM layers for NeurIPS 2019 Traffic4Cast competition.
NOTE: ALTERNATIVE TO BASELINE_SEQ2SEQ.PY, WHERE THE SAME MODEL IS TRAINED ONLY FOR THE PREDICTION 
TIMES AS DEFINED IN THE TEST DATA SETS.

This script serves for training a baseline model with Traffic4Cast data for one city.
It includes methods for:
    - loading and preprocessing the training data for one day
    - building the model in keras
    - training the model and saving its parameters

**Summary**

*Model Structure*

- the expected inputs are of shape (None, 3, 3, 495, 436), referring to [t-3:t] time bin 
    arrays previous to the prediction time t with 3 channels, 495 rows and 436 columns 
- a two layer convLSTM serves as encoder, of which the memory state of the final cell are 
    repeated three times (corresponding to the prediction horizon of 5, 10, and 15 minutes). 
- a two layer convLSTM decoder produces the final predictions in the shape of (None, 3, 3, 495, 436).

*Data Loading and Preprocessing*

- the hdf5 file names in the training data directory are listed and filtered for used 
    defined excluded dates (e.g., holidays, not used here)
- from the data, only the intervals [t-3 : t] before each t in indices are extracted, leading to
    5 samples of 6 time bins each (3 as input, 3 as ground truth)
- it is then transposed to the expected input shape of the model, and rescaled to 0-1.

*Model Training*

- the model is built and compiled with MSE loss and the ADAM optimizer (initial learning rate of 0.0001)
- each training epoch, the data for one day is loaded and preprocessed, and the model is 
    trained using a batch size of 1. This process is repeated for all days before the next epoch begins.
- the loss is logged after each iteration and the model parameters are saved after each epoch.

"""

import numpy as np
import h5py
import os, csv, re, datetime
import sys, getopt
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import backend as K
from tensorflow import device

#set indices (correspond to prediction times in test sets)
utcPlus2 = [30, 69, 126, 186, 234]
utcPlus3 = [57, 114, 174, 222, 258]
#Data Loader Functions

def load_data(file_path, indices):
    """Load data for one test day, return as numpy array with normalized samples of each
        6 time steps in random order.
    
        Args.:
            file_path (str): file path of h5 file for one day
            indices (list): list with prediction times (as list indices in the interval [0, 288])
            
        Returns: numpy array of shape (5, 6, 3, 495, 436)
    """
    #load h5 file
    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = list(fr[a_group_key])
    
    #identify test cases and split in samples of each length 3 time bins
    data = [data[y - 3 : y + 3] for y in indices]
    data = np.stack(data, axis=0)
    
    #transpose to (samples, timesteps, channels, rows, columns)
    data = np.transpose(data, (0, 1, 4, 2, 3))
    
    #rescale and return data
    data = data.astype(np.float32)
    np.random.shuffle(data)
    data /= 255.
    return data


def list_filenames(directory, excluded_dates):
    """Auxilliary function which returns list of file names in directory in random order, 
        filtered by excluded dates.
    
        Args.: 
            directory (str): path to directory
            excluded_dates (list): list of dates which should not be included in result list, 
                e.g., ['2018-01-01', '2018-12-31']
        
        Returns: list
    """
    filenames = os.listdir(directory)
    np.random.shuffle(filenames)
        
    #check if in excluded dates
    excluded_dates = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in excluded_dates]
    filenames = [x for x in filenames if return_date(x) not in excluded_dates]
    return filenames


def return_date(file_name):
    """Auxilliary function which returns datetime object from Traffic4Cast filename.
    
        Args.:
            file_name (str): file name, e.g., '20180516_100m_bins.h5'
        
        Returns: date string, e.g., '2018-05-16'
    """
    
    match = re.search(r'\d{4}\d{2}\d{2}', file_name)
    date = datetime.datetime.strptime(match.group(), '%Y%m%d').date()
    return date


#Model Definition
    
def build_model():
    """Build keras model.
    
        Returns: keras model
    """
    
    #define model input
    data_shape = (3, 3, 495, 436)
    prev_frames = layers.Input(shape=data_shape, name='prev_frames')
    
    #define layers
    with device('/device:GPU:0'):
        convlstm_0 = layers.ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=True, return_state=False,
                             activation='tanh', recurrent_activation='hard_sigmoid',
                             kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                             data_format='channels_first', name='convlstm_0')
        
        convlstm_1 = layers.ConvLSTM2D(filters=64, kernel_size=(7, 7), padding='same', return_sequences=False, return_state=True,
                             activation='tanh', recurrent_activation='hard_sigmoid',
                             kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                             data_format='channels_first', name='convlstm_1')
    
    with device('/device:GPU:1'):
        convlstm_2 = layers.ConvLSTM2D(filters=64, kernel_size=(7, 7), padding='same', return_sequences=True, return_state=False,
                             activation='tanh', recurrent_activation='hard_sigmoid',
                             kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                             data_format='channels_first', name='convlstm_2')
        
        convlstm_3 = layers.ConvLSTM2D(filters=3, kernel_size=(7, 7), padding='same', return_sequences=True, return_state=False,
                             activation='tanh', recurrent_activation='hard_sigmoid',
                             kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                             data_format='channels_first', name='convlstm_3')
    
        
    #define model structure  
    #encoder      
    x = convlstm_0(prev_frames)
    x = convlstm_1(x)[-1]
    
    #flatten, repeat and reshape
    x = layers.Flatten()(x)
    x = layers.RepeatVector(3)(x)
    x = layers.Reshape((3, 64, 495, 436))(x)
    
    #decoder
    x = convlstm_2(x)
    x = convlstm_3(x)
            
    #build and return model
    seq_model = models.Model(inputs=prev_frames, outputs=x)
    return seq_model

#Training Procedure

def model_train(data_dir, model_dir, log_path, indices, excl_dates=[]):
    """Build model, load data and train. Save model after each training epoch.
    
        Args.: 
            data_dir (str): path to directory with training data
            model_dir (str): path to directory to store model
            log_path (str): path to log file
            indices (list): list with prediction times (as list indices in the interval [0, 288])
            excluded_dates (list): list of dates which should not be included in result list, e.g., 
                ['2018-01-01', '2018-12-31']
        
        Returns: None
    """
    #generate log file
    log_file = open(log_path, 'w', newline='' )
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'file', 'loss'])
    
    #build and compile model
    model = build_model()
    optimizer = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss=losses.mean_squared_error)
    
    #receive filenames of training days
    file_names = list_filenames(data_dir, excl_dates)
    
    #train model for each training data set
    epochs = 50
    for e in range(epochs):
        for f in file_names:
            data = load_data(data_dir + f, indices)
            
            #define first 3 time bins as input, last 3 as labels
            x = data[:, :3]
            y = data[:, 3:]
        
            #train for mini-batches
            with device('/device:GPU:3'):
                hist = model.fit(x, y, epochs=1, batch_size=3)
                loss = hist.history["loss"]
            
            #log loss
            log_writer.writerow([e, f, loss[0]])
            log_file.flush()
            
        #save model
        model.save(model_dir + r'model_ep_{}.h5'.format(e))
        
#Run Model
if __name__ == '__main__':

    data_dir = ''
    model_dir = ''
    log_path = ''
    city = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:m:l:c:", ["data_dir","model_dir","log_path","city"])
    except getopt.GetoptError:
        print('usage: baseline_seq2seq_foc_train.py -d <data_dir> -m <model_dir> -l <log_path> -c <city>')
        sys.exit(2)
    for opt,arg in opts:
        if opt == '-h':
            print('usage: baseline_seq2seq_foc_train.py -d <data_dir> -m <model_dir> -l <log_path> -c <city>')
        elif opt in ("-d","--data_dir"):
            data_dir = arg
        elif opt in ("-m","--model_dir"):
            model_dir = arg
        elif opt in ("-l","--log_path"):
            log_path = arg
        elif opt in ("-c","--city"):
            city = arg
    if city in ("Berlin","Istanbul","Moscow"):
        training_file_dir = os.path.join(data_dir, city, city+"_test")
        training_indices = utcPlus3
        if city == "Berlin":
            training_indices = utcPlus2
        model_train(training_file_dir, model_dir, log_path, training_indices)
    else:
        print('invalid city provided')
        sys.exit(2)


