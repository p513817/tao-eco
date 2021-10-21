import os

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from color_log import custom_logger_with_logfile as logger
import json
from PIL import Image

class preprocess_tools:
    
    def resize_short(self, img, debug=False):
        
        h, w, c = img.shape
        scale = 255/(h if h<w else w )
        res = cv2.resize(img, ( int(w*scale), int(h*scale)))
        
        if debug: print('After Resize, Image Shape: {}'.format(res.shape))
        return res

    def crop_center(self, img, size=224, debug=False):

        h, w, c = img.shape
        start_x, start_y = (w//2-(size//2)), (h//2-(size//2))
        res = img[start_y:start_y+size, start_x:start_x+size]
        if debug: print('After Crop, Image Shape: {}'.format(res.shape))
        return res

    def subtract_avg(self, img):
        img = img.astype(np.float32)
        
        for i in range(3):
            avg=np.average(img[:,:,i])
            
            img[:,:,i]-=avg

        return img

    def caffe_mode(self, img, test=False):
        
        # size = (3,224,224)
        # 1. reshape to 256 , depend on short side
        # 2. Crop center to 224
        # 3. Subtract Average ( uint8 -> fp32)
        # 4. HWC -> CHW

        img_reshape = self.resize_short(img)
        
        img_crop = self.crop_center(img_reshape)

        img_avg = self.subtract_avg(img_crop)
        
        img_chw = img_avg.transpose( (2, 0, 1) )    

        return img_chw 

    def caffe_mode_backup(self, img, test=False):
        
        size = 224
        # reshape to 256 , depend on short side
        img_reshape = self.resize_short(img, debug=True)
        # Crop center to 224
        img_crop = self.crop_center(img_reshape, debug=True)
        # HWC -> CHW
        img_chw = img_crop.transpose( (2, 0, 1) )    
        # Normalize
        img_norm = img_chw/255  

        return img_norm           

######################################################

def process_image(arr, w=224, h=224):
    image = Image.fromarray(np.uint8(arr))

    image_resized = image.resize(size=(w, h), resample=Image.BILINEAR)
    img_np = np.array(image_resized)
    # HWC -> CHW
    img_np = img_np.transpose((2, 0, 1)).astype(trt.nptype(trt.float32))
    # Normalize to [0.0, 1.0] interval (expected by model)
    img_np = (1.0 / 255.0) * img_np
    print(img_np.shape)
    img_np = img_np.ravel()
    return img_np

class map2class:

    def __init__(self, json_path):
        
        # Check & Load Json File
        if not os.path.exists(json_path):
            logger().error("dataset.json is not exists")
            exit()
        else:
            with open(json_path) as jsonfile:
                self.classmap = json.load(jsonfile)

        # Make a mapping list
        self.class_name, self.class_idx = [], []
        for name, val in self.classmap.items():
            self.class_name.append(name)
            self.class_idx.append(val)

    def get_name(self, idx):
        return self.class_name[idx] if idx in self.class_idx else None

# Load TensorRT Engine
def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        # Close Max_Batch Size
        size = trt.volume(engine.get_binding_shape(binding)) #* engine.batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
