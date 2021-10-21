#!/usr/bin/python3
import tensorrt as trt
import utils
from color_log import custom_logger_with_logfile as logger
import os
import numpy as np
import time
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("engine", help="Path of TensorRT Engine.")
parser.add_argument("input", help="Path of Input data ( image, video , image folder or camera:/dev/video* )")
parser.add_argument("json", help="Json file which was built by tao toolkit ( classmap.json )")
args = parser.parse_args()

#######################################################

mode = ''   
video_ext = ['.mp4']
image_ext = ['.jpg', '.png', '.jpeg']

if os.path.isdir(args.input):
    mode = 'dir'
elif 'video' in args.input:
    mode = 'cam'
else:
    root, ext = os.path.splitext(args.input)
    mode = 'vid' if ext in video_ext else 'img'
        
trt_engine_path = args.engine
input_path = args.input

# Define Log
log = logger()
# Define pre-process-tools
preprocess_tools = utils.preprocess_tools()
# Define map-tools
map2class = utils.map2class(args.json)

#######################################################

log.info("Initial TensorRT Engine")

# Initial
t_start = time.time()
trt_engine_datatype = trt.DataType.FLOAT
batch_size = 1

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# We first load all custom plugins shipped with TensorRT,
# some of them will be needed during inference
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

# Initialize runtime needed for loading TensorRT engine from file
trt_runtime = trt.Runtime(TRT_LOGGER)
# TRT engine placeholder
trt_engine = None

# Display requested engine settings to stdout
log.info("TensorRT inference engine settings:")
log.info("  * Inference precision - {}".format(trt_engine_datatype))
log.info("  * Max batch size - {}".format(batch_size))

# If we get here, the file with engine exists, so we can load it
if not trt_engine:
    log.info("Loading cached TensorRT engine from {}".format(trt_engine_path))
    trt_engine = utils.load_engine(
        trt_runtime, trt_engine_path)

# This allocates memory for network inputs/outputs on both CPU and GPU
inputs, outputs, bindings, stream = utils.allocate_buffers(trt_engine)

# Execution context is needed for inference
context = trt_engine.create_execution_context()

# # Allocate memory for multiple usage [e.g. multiple batch inference]
# input_volume = trt.volume(ModelData.INPUT_SHAPE)
# numpy_array = np.zeros((trt_engine.max_batch_size, input_volume))
# log.info("Initial TensorRT Engine Finished ({:.3f})s".format(time.time()-t_start))

#######################################################

if mode=='img' or mode=='dir':

    log.warning("Image Mode")

    if os.path.isdir(input_path):
        log.warning("Path is directory")
        images = [ os.path.join(input_path, img) for img in os.listdir(input_path) ]
    else:
        images = [ input_path ]

    log.info("Load TensorRT Engine")

    for image in images:
        
        if not os.path.splitext(image)[1] in image_ext: continue

        img = preprocess_tools.caffe_mode(cv2.imread(image))

        np.copyto(inputs[0].host, img.ravel())

        inference_start_time = time.time()

        # Fetch output from the model
        [res] = utils.do_inference(
            context, bindings=bindings, inputs=inputs,
            outputs=outputs, stream=stream)

        # Parse Results
        idx = res.argmax()
        name = map2class.get_name(idx)
        prop = res[idx]

        # Show Results
        content = "Input: {}, Prediction: {} ({:.3f})".format(image, name, prop)
        log.info(content)

        # Output inference time
        log.info("TensorRT inference time: {} ms\n".format(
            int(round((time.time() - inference_start_time) * 1000))))

elif mode=='cam':
    
    log.warning("Stream Mode")

    cap = cv2.VideoCapture(args.input)

    log.warning("Press 'q' to quit")

    while(cap.isOpened() ):

        t_start = time.time()
        
        ret, frame = cap.read()

        if frame is None: continue

        img = preprocess_tools.caffe_mode(frame)

        np.copyto(inputs[0].host, img.ravel())

        inference_start_time = time.time()

        # Fetch output from the model
        [res] = utils.do_inference(
            context, bindings=bindings, inputs=inputs,
            outputs=outputs, stream=stream)

        # Parse Results
        idx = res.argmax()
        name = map2class.get_name(idx)
        prop = res[idx]

        # Show Results
        content = "FPS: {:.2f}, Prediction: {:<5} ({:.3f})".format(1/(time.time()-t_start), name, prop)

        log.info(content)
        cv2.putText(frame, content, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, content, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow('USB Detector', frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

elif mode=='vid':

    log.error("still not support")
    exit()

log.warning("QUIT")
