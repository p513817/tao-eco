#!/usr/bin/python3
import cv2
import json
import os
import numpy as np
import time
import argparse
from color_log import custom_logger_with_logfile as logger
from color_log import bcolors
import shutil

def get_total(path):
    return len(os.listdir(path))

def get_info(frame, classes, info, nums=None):
    
    h,w,c = frame.shape
    step = 60
    h = len(classes)*step

    info_bg = np.ones( (h, w, 3))

    start_x = 5
    start_y = 20
    step_y = 25
    for i, line in enumerate(info.split('\n')):

        y = start_y + i*step_y
        # cv2.putText(info_bg, line, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA )
        cv2.putText(info_bg, line, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA )

    return info_bg

def clear_data(class_path):
    
    for classes in class_path.values():
        try:
            shutil.rmtree(classes)
            os.makedirs(classes)
        except:
            pass

def update_info(class_path):
    info = ''
    for idx,classes in class_path.items():
        # Setup Info
        info += f'Press [{idx}] to save in {classes}, nums:{get_total(classes)} \n'
        # Map to ord
        key_class.append(ord(idx))
        
    info += f'Press [q] to leave\n\n'
    return info
    

log = logger()
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--id", default=0, help="ID of Camera, e.g. /dev/video*, * is your ID")
parser.add_argument("-j", "--json", default="./dataset.json", help="Path of dataset.json")
args = parser.parse_args()

log.info("Initial")
WIN_MAIN="Make Dataset"
WIN_SUB="Info"
cv2.namedWindow(WIN_MAIN)
cv2.namedWindow(WIN_SUB)

class_path = {}
img_name = {}
key_class = []

if not os.path.exists(args.json):
    log.error("dataset.json is not exists")
    exit()
else:
    # Load Json File
    with open(args.json) as jsonfile:

        cnt = json.load(jsonfile)
        
        # Check Directory and Save to class_path
        path, classes = cnt["path"], cnt["classes"]        
        
        for idx,cls in enumerate(classes):
            
            trg_path = os.path.join(path, cls)
            
            class_path[str(idx)] = trg_path # for path dict
            img_name[str(idx)] = 0  # for name list

            if not os.path.exists(trg_path):
                os.makedirs(trg_path)
                log.warning("Create Class Path: {}".format(trg_path))
    
    # Set Information Content
    info = update_info(class_path)

cap = cv2.VideoCapture(args.id)

while(cap.isOpened()):
    
    ret, frame = cap.read()

    if frame is None: continue

    win_info = get_info(frame, class_path, info)

    # Show Windows
    win_main_pos = [50, 50]
    win_info_pos = [win_main_pos[0]+frame.shape[1] ,50]
    cv2.moveWindow(WIN_MAIN, 50, 50)
    cv2.moveWindow(WIN_SUB, 50+frame.shape[1], 50)

    cv2.imshow(WIN_SUB, win_info)
    cv2.imshow(WIN_MAIN, frame )
    key = cv2.waitKey(1)

    if key in key_class:
        # Get path and img name        
        idx = str(key_class.index(key))

        img_path = os.path.join( class_path[idx], f'{img_name[idx]}.jpg')

        cv2.imwrite(img_path, frame)

        img_name[idx] +=1

        info = update_info(class_path)
    elif key ==ord('c'):
        log.warning("Remove Data")
        clear_data(class_path)
        info = update_info(class_path)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

log.warning("QUIT")
