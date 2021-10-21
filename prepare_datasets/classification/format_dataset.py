#!/usr/bin/python3
import os
import numpy
import logging
import shutil
import argparse
from random import shuffle
from tqdm import tqdm
import glob

"""
DIR : data
NAME: defect, perfect
SAMPLE: ./data/defect.Z252_3.jpg

1. prepare label dir and classify usb data from train and test

- formatted
    - label_1_name
        - *.jpg
    - label_2_name
        - *.jpg

2. split train/val/test

I will keep train and test , val will split from train directory.
- split
    - train
    - val
    - test    
"""
# Argument Setting
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--format", action="store_true", help="format data by name")
parser.add_argument("-s", "--split", action="store_true", help="split 'formatted' directory to ./split/train, ./split/val, ./split/test")

parser.add_argument("--froot", help="if you only need to format")
parser.add_argument("--sroot", help="if you only need to split, you can use froot to split your dataset")
args = parser.parse_args()

# Initialize
logging.basicConfig(level=logging.DEBUG)
logging.info('Initialize ... ')



if args.format:

    logging.error("Broken Function")
    exit(1)

    ## Original Path
    root = os.path.abspath(args.froot)

    data_dir = os.path.join(root, 'data')

    if os.path.exists(data_dir):
        logging.info('Find data directory')
    else:
        logging.error('Can not find data directory')
        exit(1)

    ## Formatted Path
    formatted_dir = os.path.join(root, 'formatted')
    perfect_dir = os.path.join(formatted_dir, 'perfect')
    defect_dir = os.path.join(formatted_dir, 'defect')

if args.split:

    ## Split Path
    formatted_dir = args.sroot
    split_path = os.path.join('./', 'split')
    train_dir = os.path.join(split_path, 'train')
    test_dir = os.path.join(split_path, 'test')
    val_dir = os.path.join(split_path, 'val')

    ## Check and Create All Directory
    logging.info('Update split folder ... ')
    list_dir = [train_dir, test_dir, val_dir ]

    for d in list_dir:
        if os.path.exists(d):        
            shutil.rmtree(d)
            os.makedirs(d)
            logging.info(f'Clear directory ({d})')  
        else:
            os.makedirs(d)
            logging.info(f'Create directory ({d})')

# if args.format and args.split:

#     ## Check and Create All Directory
#     logging.info('Checking All Directory ... ')
#     list_dir = [perfect_dir, defect_dir, train_dir, test_dir, val_dir ]
#     for d in list_dir:
#         if not os.path.exists(d):
#             os.makedirs(d)
#             if os.path.exists(d): logging.info(f'Create directory ({d})')

# Main
## Format For Custom Data
if args.format:

    data = os.listdir(data_dir)
    logging.info('Get {} data in {}'.format(len(data), data_dir))
    error_log = []

    logging.info('Format custom data is starting')
    for f in tqdm(data):
        file_path = os.path.join(data_dir, f)
        f_parsed = f.split('.')
        if len(f_parsed)==3:
            label, name, ext = f_parsed
            shutil.copy2(file_path, perfect_dir if label == 'perfect' else defect_dir)
        else:
            error_log.append(f)
        
    logging.warning('Found wrong format file:')
    for log in error_log: print(log, '\t')

    logging.info('Format custom data finished')

    ## Checking Format Data
    logging.info('Get {} data in {}'.format(len(os.listdir(perfect_dir)), perfect_dir))
    logging.info('Get {} data in {}'.format(len(os.listdir(defect_dir)), defect_dir))

## Split Custom Data From formatted
if args.split:

    logging.info('Split custom data is starting')
    
    for label_dir in tqdm(os.listdir(formatted_dir)):
        
        # Get Full Path
        trg_label_dir = os.path.join(formatted_dir, label_dir)

        # Cteate Target Dir 
        trg_train_dir = os.path.join(train_dir, label_dir)
        trg_val_dir = os.path.join(val_dir, label_dir)
        trg_test_dir = os.path.join(test_dir, label_dir)

        for trg_dir in [trg_train_dir, trg_val_dir, trg_test_dir]:
            if not os.path.exists(trg_dir):
                os.makedirs(trg_dir)

        # Get All Images in label_dir
        img_list = glob.glob( os.path.join(trg_label_dir, '*.jpg' ))
        shuffle(img_list)    # Shuffle data

        # Split train/val/test with 7/1/2
        range_total = len(img_list)
        range_1 = int(range_total*0.7)
        range_2 = int(range_total*0.8)
        
        for idx in range(range_1):
            shutil.copy2(img_list[idx], trg_train_dir)
        for idx in range(range_1, range_2):
            shutil.copy2(img_list[idx], trg_val_dir)
        for idx in range(range_2, range_total):
            shutil.copy2(img_list[idx], trg_test_dir)            
        
    logging.info('Split data finished')

    for task in os.listdir(split_path):
        task_dir = os.path.join(split_path, task)
        task_data = 0
        for label in os.listdir(task_dir):
            label_dir = os.path.join(task_dir, label)
            task_data += len(os.listdir(label_dir))
        logging.info("Found {} data in {}".format(task_data, task_dir)) 
        
logging.info("All Done.")
