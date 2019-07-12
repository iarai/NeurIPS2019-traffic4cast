#!/usr/bin/env/python3
"""
This script clones the submission folder structure and the file names it finds in an input folder assumed of the same
format and writes out corresponding files in ouput format of constant value (which is also provided as an input).
Input:  -inpath = path to input directory (which contains 3 subdirectories named after the 3 cities each of which 
                  contains again the city name followed by _test as a folder. In it, it has zipped h5 files.
        -outpath= path where the output is written to.
        -outvalue=constant value the output file will carry in all entries or the word "random".

"""
import numpy as np
import os, csv
import sys, getopt
import h5py

cities = ['Berlin','Istanbul','Moscow']

def list_filenames(directory):
    filenames = os.listdir(directory)
    return filenames

def write_data(data, filename):
    """
    write data in gzipped h5 format.
    """
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data = data, compression='gzip', compression_opts=9)
    f.close()

def create_directory_structure(root):
    """
    create competition output directory structure at given root path. 
    """
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

def write_output_files(input_path, output_path, out_data, random = False):
    """
    write outdata into each submission folder structure at out_path, cloning
    filenames in corresponding folder structure at input_path.
    """
    create_directory_structure(output_path)
    for city in cities:
        # set relevant list
        data_dir = os.path.join(input_path, city, city+'_test')
        sub_files = list_filenames(data_dir)
        for f in sub_files:
            # load data
            outfile = os.path.join(output_path, city, city+'_test',f)
            if random:
                out = np.random.randint(256, size=(5,3,495,436,3), dtype = np.dtype(np.uint8))
            else:
                out = out_data
            write_data(out, outfile)
            print("just wrote file {}".format(outfile))

            

if __name__ == '__main__':

    # gather command line arguments.
    inpath = ''
    outpath = ''
    outvalue = int(0)
    out_data = np.zeros((5,3,495,436,3))
    random = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:v:", ["inpath=","outpath=","outvalue="])
    except getopt.GetoptError:
        print('usage: create_submissiontest_like -i <input directory path> -o <output path> -v <constant value or "random">')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: create_submissiontest_like -i <input directory path> -o <output path> -v <constant value or "random">')
            sys.exit()
        elif opt in ("-i","--inpath"):
            inpath = arg
        elif opt in ("-o","--outpath"):
            outpath = arg
        elif opt in ("-v","--outvalue"):
            if arg =='random':
                random = True
            else:
                outvaluestr = arg
                outvalue = np.uint8(int(outvaluestr))
                #generate output data structure
                print(outvalue)
                out_data = np.full(shape = (5,3,495,436,3), fill_value = outvalue, dtype = np.dtype(np.uint8)  )
    write_output_files(inpath, outpath, out_data, random=random)    


            
            
