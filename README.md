# Temporal Noise Reduction With CUDA Kernels.

## Overview

This project demonstrates the use of CUDA advanced libraries for temporal noise reduction for video and perform cubic spline interpolation of function Ax=B. The goal is to utilize GPU acceleration to execute kernels and perform multi frame processinig. The project is a Capstone project for Cuda Advanced Libraries and serves as an example for understanding how to use different cuda advanaced libraries for video processing.

## Project 
This project mainly focus on implementing different cuda kernels and suing advanced libraaries to perform linear algaebra functions or using tensor libraries to generate tensors and peform complex convolution operations.

# Temporal Noise Reduction.
Three kernels are implemented to perform temporal noise reduction of video. The first is a simple noise reduction comparing current and previous frame and using alpha coefficient for proportioning two frames. The second kernel is windowing temporal noise reduction where sliding window with kernel dimension 5x5 is used to generate median and replace pixels of the filter set to one with that value of YUV frame. The third kernel is weighted temporal nosie reduction. It associates wieghts to pixels and average out weights for the pixels to replace the middle pixel. The CPU option run processing on the CPU while CUDA options runs kernels on GPU. 

# Cubic Spline Interpolation.
The interpolation coefficients are generated with Ax=B function where coefficient x is calculated by taking transpose of A and performing matrix multiplication with B. 
x=(A)transposeB The Cubic spline interpolation with X, Y coefficients where X corresponds to the frame number and Y is the coefficients generated with above function. 

## Code Organization

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder should hold all example data in any format. If the original data is rather large or can be brought in via scripts, this can be left blank in the respository, so that it doesn't require major downloads when all that is desired is the code/structure.

```lib/```
Any libraries that are not installed via the Operating System-specific package manager should be placed here, so that it is easier for inclusion/linking.

```src/```
The source code should be placed here in a hierarchical fashion, as appropriate.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile or CMAkeLists.txt or build.sh```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.

## Key Concepts

Performance Strategies, Image Processing, NPP Library

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, ppc64le, armv7l

## CUDA APIs involved

## Dependencies needed to build/run
[FreeImage](../../README.md#freeimage), [NPP](../../README.md#npp)

## Prerequisites

Download and install the [CUDA Toolkit 11.4](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run

### Windows
The Windows samples are built using the Visual Studio IDE. Solution files (.sln) are provided for each supported version of Visual Studio, using the format:
```
*_vs<version>.sln - for Visual Studio <version>
```
Each individual sample has its own set of solution files in its directory:

To build/examine all the samples at once, the complete solution files should be used. To build/examine a single sample, the individual sample solution files should be used.
> **Note:** Some samples require that the Microsoft DirectX SDK (June 2010 or newer) be installed and that the VC++ directory paths are properly set up (**Tools > Options...**). Check DirectX Dependencies section for details."

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
```
The samples makefiles can take advantage of certain options:
*  **TARGET_ARCH=<arch>** - cross-compile targeting a specific architecture. Allowed architectures are x86_64, ppc64le, armv7l.
    By default, TARGET_ARCH is set to HOST_ARCH. On a x86_64 machine, not setting TARGET_ARCH is the equivalent of setting TARGET_ARCH=x86_64.<br/>
`$ make TARGET_ARCH=x86_64` <br/> `$ make TARGET_ARCH=ppc64le` <br/> `$ make TARGET_ARCH=armv7l` <br/>
    See [here](http://docs.nvidia.com/cuda/cuda-samples/index.html#cross-samples) for more details.
*   **dbg=1** - build with debug symbols
    ```
    $ make dbg=1
    ```
*   **SMS="A B ..."** - override the SM architectures for which the sample will be built, where `"A B ..."` is a space-delimited list of SM architectures. For example, to generate SASS for SM 50 and SM 60, use `SMS="50 60"`.
    ```
    $ make SMS="50 60"
    ```

*  **HOST_COMPILER=<host_compiler>** - override the default g++ host compiler. See the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements) for a list of supported host compilers.
```
    $ make HOST_COMPILER=g++
```


## Running the Program
After building the project, you can run the program using the following command:

```bash
Copy code
./run.sh
```

This command will execute the compiled binary and save the result as <image>.exr in the data/ directory. 

To run cubic spline interpolation inerpolation. The coefficients are populated the code. 
```bash
Copy code
python3 ./src/interpolation.py
```

If you wish to run the binary directly with custom input/output fies for CUDA, you can use:

```bash
- Copy code
./bin/TemporalNoiseReductionCuda 20063221-uhd_3840_2160_24fps.mp4 CUDA
```
If you wish to run the binary directly with custom input/output fies for CPU, you can use:

```bash
- Copy code
./bin/TemporalNoiseReductionCuda 20063221-uhd_3840_2160_24fps.mp4 CUDA
```

- Cleaning Up
To clean up the compiled binaries and other generated files, run:

```bash
- Copy code
make clean
```
This will remove all files in the bin/ directory.
