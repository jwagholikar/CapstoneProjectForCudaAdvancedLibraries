# Temporal Noise Reduction With CUDA Kernels.

## Overview

This project demonstrates the use of CUDA advanced libraries for temporal noise reduction for video and perform cubic spline interpolation of function Ax=B. The goal is to utilize GPU acceleration to execute kernels and perform multi frame processing. The project is a capstone project for cuda advanced libraries and serves as an example for understanding how to use different cuda advanaced libraries for video processing.

## Project 
This project mainly focus on implementing different cuda kernels and suing advanced libraaries to perform linear algaebra functions or using tensor libraries to generate tensors and peform complex convolution operations.

# Temporal Noise Reduction.
There are total three kernels are implemented for denpising the video. The input vidoe is converted from BGR to YUV plane and in NV12 format for faster computation. The data is also generated for denoising of grey frames. The first kernel mainly helps reducing noise by taking percentage of current and previous pixels values based on the defined alpha coefficient. If alpha coefficient is 0.75 then 75% of current frame pixels intesity is added to 25% of previous frame pixel intesity to generate a new frame. The second kernel mainly works as a sliding window on the given frame based on $3x3$, $5x5$ or $7x7$ filter. Based on the filter values the median of the pixel intensity is calculated on the filtered image and applied to the filtered frame to denoise pixel intensity. The third kernel is weighted temproal reductions where weights are calculated per pixel intensity and average weight is assigned to current frame. In YUV plan the weight associated with Y plane pixels is twice the weight associated with UV plane pixels.

# Cubic Spline Interpolation.
Normally directly interpolating noisy data can amplify noisy data. The interpolation coefficients are generated based on the noised and denoised frames. The function used for it is $Ax=B$ where x is unknown coefficient and its value is calculated by perforining matrix multiplication of $A^TB$ These coeffcients are used for a given time $t$ to calculate final coefficient c(t) as $c(t) = H3,0(t) * p0 + H3,1(t) * v0 + H3,2(t) * v1 + H3,3(t) * p1$ The cubic spline interpolation of these coefficients are plotted by generating new X values spread across MIN and MAX range. GEMM operator can be used to perform matrix multiplication operation for larger frames. There can be limited memory associated for allocation of larger matrix so coefficients are calculated for sample video frames and plotted against newly generated X values. This feature is disabled right now.The data points for the plot are already used from the extracted data.

# Performance
The kernel performance is measured on Nvidia A10G GPU and CPU. 
| MP4 Video Input | Kernel 0 GPU | Kernel 1 GPU | Kernel 2 GPU | CPU  | Percentage Improvement | 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| hd_1920_1080_30fps | 6.091455 ms | 697.605835 ms | 64.703247 ms |	1077526.375000 ms | 93.5%  |
|uhd_3840_2160_24fps | 9.569598 ms | 1095.937622 ms | 92.275208 ms | 1881330.375000 ms | 99.41% |
|hd_1920_1080_30fps  | 49.725464 ms | 695.662598 ms	| 63.530586 ms | 1028540.750000 ms | 99.93% |

# Memory Usage
Using nvidia-smi command
| Process Id | Process Name | Used GPU Memory[MiB] | 
| ------------- | ------------- | ------------- |
| 2703050 | TemporalNoiseReduction| 528 MiB |
| 2733958  | TemporalNoiseReductionKernel |  248 MiB |

Using cudaMemGetInfo()

CUDA Device Memory Information:

Total Memory: 22617.8 MB

Free Memory:  21626.1 MB

Used Memory:  991.688 MB

## Code Organization

```bin/```
The executable is present in the bin directory. 

```data/```
Since video data is in megabytes it's available in the data link directory.

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
