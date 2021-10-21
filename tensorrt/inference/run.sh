#!/bin/bash
./trt_infer.py ../max_other_detect/max_other_gtx1050_e100.trt /dev/video0 -j ../max_other_detect/classmap.json 
