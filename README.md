TensorFlow_CarOccupancyDetector_V2
TensorFlow - Car-Occupancy Detector/Classifier from Scratch


THIS is now Version 2 of this Solution:

Smart City How might we create smart solutions to cope with the increasing individual car traffic in our cities?
https://github.com/iCounterBOX/Themes-from-INTEREST/wiki/Smart-City---How-might-we-create-smart-solutions-to-cope-with-the-increasing-individual-car-traffic-in-our-cities%3F

Many thanks to this guy ( edje ) / https://www.youtube.com/watch?v=Rgpfk6eYxJA   and https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

Mainly i followed this Youtube video...but here and there i struggled with different Version conflicts on TF, CUDA, cuDNN, Path-conflicts ..etc. I am quite new on this Stuf - SO this "Experiance-BLOG" is also a view out from a Newbee perspective.

Main Goal:

My main target is still to detect "HOW many People are sitting in a Car?". At the via WebCam i will get alerted if ONE or TWO Persons are detected.

Step by Step:

My Setup

PC: ACER NITRO 5 /  Win 10 / 32 GB RAM / GeForce GTX 1660 Ti


![image](https://user-images.githubusercontent.com/37293282/67181048-bca79f00-f3db-11e9-87b6-22c902c587a6.png)


I made a fresh Installation of Anaconda3 & CuDnn & CUDA / Gforce

Dependency-List: https://www.tensorflow.org/install/source_windows

I decided to setup.. tensorflow_gpu-1.12.0 + Python 3.6.5 + cuDNN 7 + CUDA9

Anaconda3 (https://www.anaconda.com/)
I use Anaconda and Spyder and the Anaconda-Prompt! My first try to setup ALL within Anaconda-GUI ( TF, Cuda,..) did not work – the different dependent Versions made conflicts. 
At the end I setup “SOME” in anaconda and some  within the environment-Prompt – figuring out the proper Versions was the biggest pain..


CUDA-Toolkit
For some Tests i need VS2017 - this was installed on my PC before!
Mainly i followed Marc Jay https://www.youtube.com/watch?v=Ebo8BklTtmc).

..here are some remarks i noticed during this installation:
I take the Gforce-Experience 
![image](https://user-images.githubusercontent.com/37293282/67181560-35f3c180-f3dd-11e9-9b0a-a22022c54186.png)
to setup and Install the Latest drivers (https://www.nvidia.com/content/DriverDownload-March2009/confirmation.php?url=/Windows/436.48/436.48-notebook-win10-64bit-international-whql.exe&lang=us&type=geforcem)


