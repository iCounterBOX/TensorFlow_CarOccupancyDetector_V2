# TensorFlow_CarOccupancyDetector_V2  

TensorFlow - Car-Occupancy Detector/Classifier from Scratch


THIS is now Version 2 of this Solution:

S*mart City How might we create smart solutions to cope with the increasing individual car traffic in our cities?
https://github.com/iCounterBOX/Themes-from-INTEREST/wiki/Smart-City---How-might-we-create-smart-solutions-to-cope-with-the-increasing-individual-car-traffic-in-our-cities%3F*

Many thanks to this guy ( edje ) / https://www.youtube.com/watch?v=Rgpfk6eYxJA   and https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

Mainly i followed this Youtube video...but here and there i struggled with different Version conflicts on TF, CUDA, cuDNN, Path-conflicts ..etc. I am quite new on this Stuf - SO this "Experiance-BLOG" is also a view out from a Newbee perspective.

**Main Goal:**

Main target is still to detect "**HOW many People are sitting in a Car?**". Via WebCam i will get alerted if ONE or TWO Persons are detected.

Step by Step:

My Setup: PC: ACER NITRO 5 /  Win 10 / 32 GB RAM / GeForce GTX 1660 Ti

I made a fresh Installation of Anaconda3 & CuDnn & CUDA / Gforce

Dependency-List: https://www.tensorflow.org/install/source_windows


![image](https://user-images.githubusercontent.com/37293282/67208929-b8e53e00-f416-11e9-927f-cc55e9b7daa4.png)

I decided to setup.. **tensorflow_gpu-1.12.0 + Python 3.6.5 + cuDNN 7 + CUDA9**

**Anaconda3** (https://www.anaconda.com/)  
I use Anaconda and Spyder and the Anaconda-Prompt! My first try to setup ALL within Anaconda-GUI ( TF, Cuda,..) did not work – the different dependent Versions made conflicts. 
At the end I setup “SOME” in anaconda and some  within the environment-Prompt – figuring out the proper Versions was the biggest pain..

IN Anaconda-GUI i create the environment   
**tfgpu112**  
This env was created with  **Phython 3.6** ( choosen from DropDown )

**CUDA-Toolkit**

For some Tests i need VS2017 - this was installed on my PC before!  
Mainly i followed Marc Jay https://www.youtube.com/watch?v=Ebo8BklTtmc).
![image](https://user-images.githubusercontent.com/37293282/67181048-bca79f00-f3db-11e9-87b6-22c902c587a6.png)

..here are some remarks i noticed during this installation:
I take the Gforce-Experience 
![image](https://user-images.githubusercontent.com/37293282/67181560-35f3c180-f3dd-11e9-9b0a-a22022c54186.png)
to setup and Install the Latest drivers (https://www.nvidia.com/content/DriverDownload-March2009/confirmation.php?url=/Windows/436.48/436.48-notebook-win10-64bit-international-whql.exe&lang=us&type=geforcem)


edje ( https://www.youtube.com/watch?v=Rgpfk6eYxJA )  is running and placing all the code for the detection under and into the folder structure from the Tensor-Flow-Installation!  
I decided to store those Code into a different folder! This lets Tesorflow as it is on HardDrive. The experimental code is located in my project-Folder..e.g.:


![image](https://user-images.githubusercontent.com/37293282/67210345-39a53980-f419-11e9-870e-27d1c9db4f27.png)

**Installations made in Anaconda-GUI** ( environment / tfgpu112 ):

Opencv 
matplotlib 
pillow  
numpy

**Installations made in Anaconda-Prompt** ( environment / tfgpu112 ):


```python
(
 Check python --version..if wrong then install this one
 conda install python==3.6.5  ( überprüfen  wichtig!)
)
conda install tensorflow-gpu==1.12.0
conda install spyder
conda install pytorch torchvision -c pytorch
```
  

From this guy ( thx 2 H. Singh / https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc) we get the hint to use as much as possible from the anaconda-gui..The idea is that A-GUI is checking the heavy Version-Dependencies for us..In some case a nice thing and i use as much as possible. BUT in some case we need specific versions.In that case i had good success with those CUDA setup in A-Prompt. 

also in his blog - a nice quick test if Tensorflow is basically working together with my GPU:

```python
import tensorflow as tf  
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

My biggest pain and reason why this training-Pipeline did not work run was based on non-compatible Versions of the installed apps we need here! This way i wrote this little Version-Checker:

```python
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
```


After following the videos, blogs and hints THIS now the configuration which finally let the training-Session RUN ( from the Version-Checker above ):

```python
Python 3.6.5 |Anaconda, Inc.| (default, Mar 29 2018, 13:32:41) [MSC v.1900 64 bit (AMD64)]
Type "copyright", "credits" or "license" for more information.
IPython 7.8.0 -- An enhanced Interactive Python.
runfile('D:/ALL_DEVEL_VIP/ALL_python/edje_objDetClassifier/getSomeVersionInfos_V1.py', wdir='D:/ALL_DEVEL_VIP/ALL_python/edje_objDetClassifier')
V E R S I O - I N F O:
OpenCV Version: 3.3.1
Python Version: 3.6.5
Numpy Version: 1.16.5
TensorFlow Version: 1.12.0
Verify GPU & cuDNN:   https://gist.github.com/wassname/34626c2d31e28ffc864fc4f3027c4489 
20191021_14-15-11
7005 cudnn
10010 cuda driver
9000 cuda compiled version
device 0: _CudaDeviceProperties(name='GeForce GTX 1660 Ti', major=7, minor=5, total_memory=6144MB, multi_processor_count=24)
```

After all this setup and specific adaptions i described ( and still following the steps from mentioned Video/Edje ) the training is running:

```python
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

# NOW the way is free to use and adapt all this for the CarOccupancy-Detection



