import platform
import numpy as np
import tensorflow as tf
import cv2   # opencv
import torch

import sys
sys.path.append("C:\\appl\\TensorFlow\\models\\research\\")
sys.path.append("C:\\appl\\TensorFlow\\models\\research\\object_detection\\utils")


print("V E R S I O - I N F O:")
print("OpenCV Version: {}".format(cv2.__version__))
print("Python Version: " + platform.python_version())
print("Numpy Version: " +  np.__version__)
print("TensorFlow Version: " +  tf.__version__)

'''
OK ABER der unten ist besser!!
#https://discuss.pytorch.org/t/pytorch-and-cuda-9-1/13126/9

print("The CUDA-VERSION: " + torch.version.cuda)


if tf.test.gpu_device_name():
 print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
 print("Please install GPU version of TF")
 
print("\nVerify GPU & Tensorflow:   https://www.codingforentrepreneurs.com/blog/install-tensorflow-gpu-windows-cuda-cudnn \n")
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
'''

print("\nVerify GPU & cuDNN:   https://gist.github.com/wassname/34626c2d31e28ffc864fc4f3027c4489 ")

import datetime
print(datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S'))


print(torch._C._cudnn_version(), 'cudnn')
print(torch._C._cuda_getDriverVersion(), 'cuda driver')
print(torch._C._cuda_getCompiledVersion(), 'cuda compiled version')
#print(torch._C._nccl_version(), 'nccl')
for i in range(torch.cuda.device_count()):
    print('device %s:'%i, torch.cuda.get_device_properties(i))
    
