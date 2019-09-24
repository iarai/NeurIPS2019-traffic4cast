#!/usr/bin/env/python3
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
This script transforms full length files in the test folders of all 3 cities into a similar
folder structure at the given output path, with the same file names but where the files are
of shape (1,5,3,495,436,3)

Input:  -inpath = path to input directory (which contains 3 subdirectories named after the 3 cities each of which 
                  contains again the city name followed by _test as a folder. In it, it has zipped h5 files.
        -outpath= path where the output is written to.

"""
import numpy as np
import os, csv
import sys, getopt
import h5py

cities = ['Berlin','Istanbul','Moscow']
# The following indicies are the start indicies of the 3 images to predict in the 288 time bins (0 to 287)
# in each daily test file. These are time zone dependent. Berlin lies in UTC+2 whereas Istanbul and Moscow
# lie in UTC+3.
utcPlus2 = [30, 69, 126, 186, 234]
utcPlus3 = [57, 114, 174,222, 258]

def list_filenames(directory):
    filenames = os.listdir(directory)
    return filenames

def load_test_file(file_path, indicies):
    """
    Given a file path, loads test file (in h5 format).
    Returns: tensor of shape (number_of_test_cases = 5, 3,495, 436 , 3) 
    """
    # load h5 file
    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = list(fr[a_group_key])

    # get relevant test cases
    data = [data[y: y+3] for y in indicies]
    data = np.stack(data,axis=0)
    # type casting
    data = data.astype(np.uint8)
    return data

def load_input_data(file_path, indicies):
    """
    Given a file path, load the relevant training data pieces into a tensor that is returned.
    Return: tensor of shape (number_of_test_cases_per_file =5, 3, 495, 436, 3)
    """
    # load h5 file into memory.
    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = list(fr[a_group_key])

    # get relevant training data pieces
    data = [data[y-3:y] for y in indicies]
    data = np.stack(data, axis=0)

    # type casting
    data = data.astype(np.float32)
    return data

def cast_moving_avg(data):
    """
    Returns cast moving average (cast to np.uint8)
    data = tensor of shape (5, 3, 495, 436, 3) of  type float32
    Return: tensor of shape (5, 3, 495, 436, 3) of type uint8
    """

    prediction = []
    for i in range(3):
        data_slice = data[:, i:]
        t = np.mean(data_slice, axis = 1)
        t = np.expand_dims(t, axis = 1)
        prediction.append(t)
        data = np.concatenate([data, t], axis = 1)

    prediction = np.concatenate(prediction, axis = 1)
    prediction = np.around(prediction)
    prediction = prediction.astype(np.uint8)

    return prediction


def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data = data, compression='gzip', compression_opts=9)
    f.close()

def create_directory_structure(root):
    berlin = os.path.join(root, "Berlin","Berlin_test")
    istanbul = os.path.join(root, "Istanbul","Istanbul_test")
    moscow = os.path.join(root, "Moscow", "Moscow_test")
    try:
        os.makedirs(berlin)
        os.makedirs(istanbul)
        os.makedirs(moscow)
    except OSError:
        print("failed to create directory structure")
        sys.exit(2)

def write_submission_files(input_path, output_path):
    create_directory_structure(output_path)
    # go through all cities
    for city in cities:
        # set relevant list
        indicies = utcPlus3
        if city == 'Berlin':
            indicies = utcPlus2
        
        # get file names
        data_dir = os.path.join(input_path, city, city+'_test')
        sub_files = list_filenames(data_dir)
        for f in sub_files:
            # load data
            data_sub = load_input_data(os.path.join(data_dir,f),indicies)
            # calculate moving average
            outdata = cast_moving_avg(data_sub)
            #generate output file path
            outfile = os.path.join(output_path, city, city+'_test',f)
            write_data(outdata, outfile)
            print("City:{}, just wrote file {}".format(city, outfile))

            

if __name__ == '__main__':

    # gather command line arguments.
    inpath = ''
    outpath = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["inpath=","outpath="])
    except getopt.GetoptError:
        print('usage: naive_baseline_mavg.py -i <input directory path> -o <output path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: naive_baseline_mavg.py -i <input directory path> -o <output path>')
            sys.exit()
        elif opt in ("-i","--inpath"):
            inpath = arg
        elif opt in ("-o","--outpath"):
            outpath = arg
    write_submission_files(inpath, outpath)    


            
            
