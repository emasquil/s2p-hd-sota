# libSGM
---
A CUDA implementation performing Semi-Global Matching.

## Introduction
---

libSGM is library that implements in CUDA the Semi-Global Matching algorithm.  
From a pair of appropriately calibrated input images, we can obtain the disparity map.

## Features
---
Because it uses CUDA, we can compute the disparity map at high speed.

## Performance
The libSGM performance obtained from benchmark sample
### Settings
- image size : 1024 x 440
- disparity size : 128
- sgm path : 4 path
- subpixel : enabled

### Results
|Device|CUDA version|Processing Time[Milliseconds]|FPS|
|---|---|---|---|
|GTX 1080 Ti|10.1|2.0|495.1|
|GeForce RTX 3080|11.1|1.5|651.3|
|Tegra X2|10.0|28.5|35.1|
|Xavier(MODE_15W)|10.2|17.3|57.7|
|Xavier(MAXN)|10.2|9.0|110.7|

## Requirements
libSGM needs CUDA (compute capabilities >= 3.5) to be installed.  
Moreover, to build the sample, we need the following libraries:
- OpenCV 3.0 or later
- CMake 3.1 or later

## Build Instructions
```
$ git clone git@github.com:ggarella/sgm_gpu.git
$ cd sgm_gpu
$ mkdir build
$ cd build
$ cmake ../  # Several options available
$ make
```

## Sample Execution
```
$ cd .../sgm_gpu
$ mkdir results
$ cd build
$ ./stereo_test <left image path format> <right image path format> <disparity_size>
left image path format: the format used for the file paths to the left input images
right image path format: the format used for the file paths to the right input images
disparity_size: the maximum number of disparities (optional)
```
"disparity_size" is optional. By default, it is 128, but can be 64, 128, 256.

```
Example of usage:
$ ./stereo_test ../images/left_gray.png ../images/right_gray.png
```

## User Manual
---
For the user manual you can consult in UserManual.pdf (in progress)


## Author
---
The "adaskit Team"  

The adaskit is an open-source project created by [Fixstars Corporation](https://www.fixstars.com/) and its subsidiary companies including [Fixstars Autonomous Technologies](https://at.fixstars.com/), aimed at contributing to the ADAS industry by developing high-performance implementations for algorithms with high computational cost.

## License
Apache License 2.0
