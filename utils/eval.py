#!/usr/bin/python3
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
This script takes a submitted path for a submission for all 3 cities and evaluates the total score.

Input:  - path to submission directory (containing Berlin/Berlin_test, Istanbul/Istambul_test and Moscow/Moscow_test as subdirectories)
        - path to golden data set
        - desired output path
        - desired output prefix

Output - a prefix.log file and a prefix.score file as well as a prefix.extended_score file logging the scores for said submission

The execpted file input format are h5 zipped files of shape (1,5,3,495,436,3), but could be of any shape.
"""
import numpy as np
import os, csv
import sys, getopt
import h5py

cities = ['Berlin','Istanbul','Moscow']


def list_filenames(directory):
    filenames = os.listdir(directory)
    return filenames

def load_test_file(file_path):
    """
    Given a file path, loads test file (in h5 format).
    Returns: tensor of shape (number_of_test_cases = 5, 3, 3, 496, 435) 
    """
    # load h5 file
    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = list(fr[a_group_key])

    # get relevant test cases
    data = [data[0:]]
    data = np.stack(data,axis=0)
    # rescale
    data = data.astype(np.float32)
    data = data/255.
    return data

def work_out_score(submission_path, golden_path, output_path, output_prefix):
    """
    Work out the relevant mse score for a given submitted unpacked test file and having the 
    path to the golden file as well.
    Assumptions: 
        - submitted file directory structure is 
            city/city_test
          for city = Berlin, Istanbul and Moscow.
        - Moreover, it assumes that the file names within the submitted directory are the same as
          in the golden file (and same as in the public file given out).
        - Assumes file types are the same as public and golden data, i.e. h5 formats (that python will read in and 
          convert to np arrays.
    Output: writes relevant files and returns overall mse value (which might be superfluous but just in case).

    """
    
    # create necessary writers:
    log_file_path = os.path.join(output_path, prefix+'.log')
    log_file = open(log_file_path, 'w')
    log_writer = csv.writer(log_file, lineterminator="\n")
    score_file_path = os.path.join(output_path, prefix+'.score')
    score_file = open(score_file_path, 'w')
    score_writer = csv.writer(score_file, lineterminator="\n")
    extended_score_file_path = os.path.join(output_path, prefix+'.extended_score')
    extended_score_file = open(extended_score_file_path, 'w')
    extended_score_writer = csv.writer(extended_score_file, lineterminator="\n")
    
    # iterate through cities.
    overall_mse = 0.0
    for city in cities:
        city_mse = 0.0
        # get file names
        data_dir_golden = os.path.join(golden_path, city, city+'_test')
        data_dir_sub = os.path.join(submission_path, city, city+'_test')
        # we assume these are now the same as the golden data set file names in the relevant directory.
        sub_files = list_filenames(data_dir_sub)
        # iterate through assumed common file names, load the data and determine the MSE and store and iterate.
        filecount = 0.0
        for f in sub_files:
            filecount = filecount + 1.0
            data_sub = load_test_file(os.path.join(data_dir_sub,f))
            data_golden = load_test_file(os.path.join(data_dir_golden,f))
            # just for debugging purposes
            # print(data_sub.shape)
            # print(data_golden.shape)
            # calculate MSE
            mse = (np.square(np.subtract(data_sub,data_golden))).mean(axis=None)
            # now book keeping.
            city_mse += mse
            log_writer.writerow([city, f, mse])
            log_file.flush()
            print("City: {} - File: {}  --- > {}".format(city, f, mse))
        city_mse /= filecount
        overall_mse += city_mse/3.0
        print(city, city_mse)
        extended_score_writer.writerow([city, city_mse])
        extended_score_file.flush()

    score_writer.writerow([overall_mse])
    score_file.flush()
    # closing all files
    score_file.close()
    extended_score_file.close()
    log_file.close()

    return overall_mse


    
# run test

if __name__ == '__main__':

    # gather command line arguments.
    golden = ''
    submitted = ''
    output = ''
    prefix = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hg:s:o:p:", ["golden=","submitted=","output=","prefix="])
    except getopt.GetoptError:
        print('usage: t4c19eval.py -g <golden file path> -s <submitted file path> -o <output file path> -p <file prefix>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: t4c19eval.py -g <golden file path> -s <submitted file path> -o <output file path> -p <file prefix>')
            sys.exit()
        elif opt in ("-g","--golden"):
            golden = arg
        elif opt in ("-s","--submitted"):
            submitted = arg
        elif opt in ("-o","--output"):
            output = arg
        elif opt in ("-p","--prefix"):
            prefix = arg
    rest = work_out_score(submitted, golden, output, prefix)
    print(rest)


            
            
