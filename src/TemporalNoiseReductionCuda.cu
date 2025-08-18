#include <cuda_runtime.h>

#include <opencv4/opencv2/core/version.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/photo/cuda.hpp>
#include <opencv4/opencv2/photo.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>

#include<fstream>
#include<iostream>
#include<vector>
#include<string>

#include <cublas.h> // For linear algebra
#include <cublas_v2.h> // For linear algebra

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


// Kernel function for temporal noise reduction
// Change this to NLM noise reduction
__global__ void weightedTemporalNoiseReductionKernel(
    char* outputFrame,
    char* currentFrame,
    char* previousFrame,
    int width,
    int height,
    float alpha,
    float sigma, int patch_size) // Weight for the current frame (0.0 - 1.0)
{
    // Calculate global thread ID
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int half_patch_size = patch_size / 2;
    float h = sigma * sigma;
    float currsum_weights = 0;
    float currsum_pixel_values = 0;

    float prevsum_weights = 0;
    float prevsum_pixel_values = 0;

    for (int dy = -half_patch_size; dy <= half_patch_size; ++dy) {
        for (int dx = -half_patch_size; dx <= half_patch_size; ++dx) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);

            float currdiff = currentFrame[ny * width + nx] - currentFrame[y * width + x];
            float prevdiff = previousFrame[ny * width + nx] - previousFrame[y * width + x];

            float currweight = expf(-(currdiff * currdiff) / h);
            float prevweight = expf(-(prevdiff * prevdiff) / h);
            
            currsum_weights += currweight;
            currsum_pixel_values += currweight * currentFrame[ny * width + nx];

            prevsum_weights += prevweight;
            prevsum_pixel_values += prevweight * previousFrame[ny * width + nx];
        }
    }

    // Ensure within image boundaries
    if (x < width && y < height) {
        // Calculate pixel index (assuming grayscale or single channel)
        int idx = y * width + x;
        outputFrame[idx] = (alpha * (currsum_pixel_values/currsum_weights)) + ((1.0f-alpha)*(prevsum_pixel_values/prevsum_weights));
    }
}

__global__ void windowingTemporalNoiseReductionKernel(
    char* outputFrame,
    char* currentFrame,
    char* previousFrame,
    int width,
    int height,
    float alpha)
{
     // Calculate global thread ID
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int windowSize = 5;

    int filter[15] {
        0, 1, 0,
        0, 1, 0,
        1, 1, 1,
        0, 1, 0,
        0, 1, 0,
    
    };

    // Increasing window size
    unsigned char currPixelValues[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    unsigned char prevPixelValues[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


    if (
        x > width - windowSize + 1 ||
        y > height - windowSize + 1 ||
        x < windowSize - 1 ||
        y < windowSize - 1
    )
    {
        return;
    }

    for (int hh = 0; hh < windowSize; hh++) 
    {
        for (int ww = 0; ww < windowSize; ww++) 
        {
            if (filter[hh * windowSize + ww] == 1)
            {
                int idx = (y + hh - 1) * width + (x + ww - 1);
                currPixelValues[hh * windowSize + ww] = currentFrame[idx];
                prevPixelValues[hh * windowSize + ww] = previousFrame[idx];

            }
        }
    }

    // Get median pixel value and assign to filteredImage
    for (int i = 0; i < (windowSize * windowSize); i++) {
	    for (int j = i + 1; j < (windowSize * windowSize); j++) {
	        if ((currPixelValues[i]) > (currPixelValues[j])) {
		        //Swap the variables.
		        char tmp = (currPixelValues[i]);
		        currPixelValues[i] = currPixelValues[j];
		        currPixelValues[j] = tmp;
	        }

            if (prevPixelValues[i] > prevPixelValues[j]) {
		        //Swap the variables.
		        char tmp = prevPixelValues[i];
		        prevPixelValues[i] = prevPixelValues[j];
		        prevPixelValues[j] = tmp;
	        }
	    }
    }

    if (x < width && y < height) {
        // Calculate pixel index 
        // calculate pixel index for y, u, v
        int idx = y * width + x;
         currentFrame[idx] = currPixelValues[(windowSize * windowSize) / 2];
         previousFrame[idx] = prevPixelValues[(windowSize * windowSize) / 2];
        // Apply weighted average
        outputFrame[idx] = static_cast<unsigned char>(
            alpha * currentFrame[idx] + (1.0f - alpha) * previousFrame[idx]
        );
    }
}

// Example: YUV420p to RGBA conversion
__global__ void YUV420pToRGBA(char* y_plane,
                              char* u_plane,
                              char* v_plane,
                              char* rgba_output,
                             int width, int height,
                             int y_pitch, int uv_pitch) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Calculate indices for Y, U, V
        int y_idx = y * y_pitch + x;
        int u_idx = (y / 2) * uv_pitch + (x / 2);
        int v_idx = (y / 2) * uv_pitch + (x / 2);

        // Get Y, U, V values
        float Y = static_cast<float>(y_plane[y_idx]);
        float U = static_cast<float>(u_plane[u_idx]);
        float V = static_cast<float>(v_plane[v_idx]);

        // Perform YUV to RGB conversion (e.g., BT.601)
        float R = Y + 1.402f * (V - 128.0f);
        float G = Y - 0.344136f * (U - 128.0f) - 0.714136f * (V - 128.0f);
        float B = Y + 1.772f * (U - 128.0f);

        // Clamp values to [0, 255]
        R = fmaxf(0.0f, fminf(255.0f, R));
        G = fmaxf(0.0f, fminf(255.0f, G));
        B = fmaxf(0.0f, fminf(255.0f, B));

        // Store RGBA output
        int rgba_idx = (y * width + x) * 4;
        rgba_output[rgba_idx] = static_cast<char>(R);
        rgba_output[rgba_idx + 1] = static_cast<char>(G);
        rgba_output[rgba_idx + 2] = static_cast<char>(B);
        rgba_output[rgba_idx + 3] = static_cast<char>(255); // Alpha
    }
}

__global__ void rgbToGreyscaleGPU(
    uchar4 *rgbImage, 
    unsigned char *greyImage,
    int rows,
    int cols
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > cols || y > rows)
    {
        return;
    }

    uchar4 rgba  = rgbImage[y * cols + x];
    unsigned char greyValue =  (0.299*rgba.x + 0.587*rgba.y + 0.114*rgba.z);
    greyImage[y * cols + x] = greyValue;
}


__global__ void simpleTemporalNoiseReductionKernel(
    char* outputFrame,
    char* currentFrame,
    char* previousFrame,
    int width,
    int height,
    float alpha)
{
     // Calculate global thread ID
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Calculate pixel index (assuming grayscale or single channel)
        int idx = y * width + x;
        
        // Apply weighted average
        outputFrame[idx] = static_cast<unsigned char>(
            alpha * currentFrame[idx] + (1.0f - alpha) * previousFrame[idx]
        );
    }
}

// Host Code to launch grayscale
void applyConversionRGBToGrey(
    uchar4 *currentFrameHost,
    unsigned char *greyFrameHost,
    int height,
    int width) 
    
{

   unsigned char *greyFrameDevice;
   uchar4 *currentFrameDevice;

   cudaMalloc(&greyFrameDevice, width * height * sizeof(unsigned char));
   cudaMalloc(&currentFrameDevice, width * height * sizeof(uchar4));

   cudaMemcpy(currentFrameDevice, currentFrameHost, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    rgbToGreyscaleGPU<<<gridSize, blockSize>>>(
        currentFrameDevice, greyFrameDevice, height, width);
    

   // Copy result back to host
    cudaMemcpy(greyFrameHost, greyFrameDevice, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(greyFrameDevice);
    cudaFree(currentFrameDevice);
}

void applyConversionYUVtoRGBA (
    std::vector<uint8_t> y_planeHost,
    std::vector<uint8_t> u_planeHost,
    std::vector<uint8_t> v_planeHost,
    unsigned char *rgbaFrameHost, 
    int width, 
    int height, 
    int y_pitch, 
    int uv_pitch) {

     // Allocate device memory for frames
    char *y_PlaneDevice;
    char *u_PlaneDevice;
    char *v_PlaneDevice;
    char *rgbaFrameDevice;

    // Allocate device memory
    cudaMalloc(&y_PlaneDevice, y_planeHost.size()*sizeof(uint8_t));
    cudaMalloc(&u_PlaneDevice, u_planeHost.size()*sizeof(uint8_t));
    cudaMalloc(&v_PlaneDevice, v_planeHost.size()*sizeof(uint8_t));
    cudaMalloc(&rgbaFrameDevice, width * height * sizeof(char));
    
    char *temp = (char*)&y_planeHost[0];
    char *temp1 =(char*)&u_planeHost[0];
    char *temp2 =(char*)&v_planeHost[0];

    // Copy host data to device
    cudaMemcpy(y_PlaneDevice, temp, y_planeHost.size()*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(u_PlaneDevice, temp1, u_planeHost.size()*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(v_PlaneDevice, temp2, v_planeHost.size()*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(rgbaFrameDevice, rgbaFrameHost, width * height * sizeof(char), cudaMemcpyHostToDevice);
  

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    // Launch Kernel
    YUV420pToRGBA<<<gridSize, blockSize>>>(y_PlaneDevice, u_PlaneDevice, v_PlaneDevice, rgbaFrameDevice, width, height, y_pitch, uv_pitch);

    // Copy result back to host
    cudaMemcpy(rgbaFrameHost, rgbaFrameDevice, width * height * sizeof(char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(y_PlaneDevice);
    cudaFree(u_PlaneDevice);
    cudaFree(v_PlaneDevice);
    cudaFree(rgbaFrameDevice);
}

// Host code to launch the kernel (simplified)
float applyTemporalNoiseReduction(
    char *outputFrameHost,
    char* currentFrameHost,
    char* previousFrameHost,
    int width,
    int height,
    float alpha,
    int kernelNum)
{
    // Allocate device memory for frames
    char* outputFrameDevice;
    char* currentFrameDevice;
    char* previousFrameDevice;

    //Measure timing information
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
     
    float sigma = 3.0f; // The h is sigma*sigma Higher value removesimage contents. 
    int patch_size=3;

    cudaError_t err= cudaMalloc(&outputFrameDevice, width * height * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc outputframe device failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaMalloc(&currentFrameDevice, width * height * sizeof(char));
     if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc currentframe device failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err =cudaMalloc(&previousFrameDevice, width * height * sizeof(char));
     if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc previousframe device failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Copy host data to device
    err = cudaMemcpy(currentFrameDevice, currentFrameHost, width * height * sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy currentframe host2device failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    err = cudaMemcpy(previousFrameDevice, previousFrameHost, width * height * sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy previousframe host2device failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    cudaEventRecord(start, 0);
    switch (kernelNum){
        case 0: 
            simpleTemporalNoiseReductionKernel<<<gridSize, blockSize>>>(
                outputFrameDevice, currentFrameDevice, previousFrameDevice, width, height, alpha);
            break;
        case 1:
             windowingTemporalNoiseReductionKernel<<<gridSize, blockSize>>>(
                outputFrameDevice, currentFrameDevice, previousFrameDevice, width, height, alpha);
            break;
        case 2: 
            weightedTemporalNoiseReductionKernel<<<gridSize, blockSize>>>(
                outputFrameDevice, currentFrameDevice, previousFrameDevice, width, height, alpha, sigma, patch_size);
            break;
        default:
            windowingTemporalNoiseReductionKernel<<<gridSize, blockSize>>>(
                outputFrameDevice, currentFrameDevice, previousFrameDevice, width, height, alpha);
            break;
    }
    
    //Stop record timing
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    
    // Synchronize device 
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE; 
    }

    // Copy result back to host
    cudaMemcpy(outputFrameHost, outputFrameDevice, width * height * sizeof(char), cudaMemcpyDeviceToHost);
     if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy outputframe device2host failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Calculate timing infomration
    // Calculate elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Free device memory
    cudaFree(outputFrameDevice);
    cudaFree(currentFrameDevice);
    cudaFree(previousFrameDevice);

    return milliseconds;     
}


float UsingOpenCvCPUDenoiseFunction(cv::Mat frame, cv::Mat outFrame) {
    int templateWindowSize=7;
    int searchWindowSize=21;
    float h=3;
    float hColor=3;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the kernel
    cudaEventRecord(start, 0);
    cv::fastNlMeansDenoisingColored(
    frame,
    outFrame,
    h,  // Adjust this value based on noise level
    hColor,
    templateWindowSize,
    searchWindowSize);

    //Stop record timing
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    // Calculate timing infomration
    // Calculate elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

   return milliseconds;

}

void extractYUVPlanes(const uint8_t* yuv_frame_data, int width, int height,
                      std::vector<uint8_t>& y_plane,
                      std::vector<uint8_t>& u_plane,
                      std::vector<uint8_t>& v_plane) {

    // Y plane
    int y_plane_size = width * height;
    y_plane.resize(y_plane_size);
    memcpy(y_plane.data(), yuv_frame_data, y_plane_size);

    // U and V planes (for YUV420p, they are quarter resolution)
    int uv_width = width / 2;
    int uv_height = height / 2;
    int uv_plane_size = uv_width * uv_height;

    u_plane.resize(uv_plane_size);
    v_plane.resize(uv_plane_size);

    // Calculate offsets
    const uint8_t* u_data_start = yuv_frame_data + y_plane_size;
    const uint8_t* v_data_start = u_data_start + uv_plane_size;

    memcpy(u_plane.data(), u_data_start, uv_plane_size);
    memcpy(v_plane.data(), v_data_start, uv_plane_size);
}

void dumpFrameToBinary(const std::string& filename, const cv::Mat& frame) {
    std::ofstream output_file(filename, std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    
    // Get the size of the raw data in bytes
    size_t dataSize = frame.total() * frame.elemSize(); 

    // Write the raw data to the file
    output_file.write(reinterpret_cast<const char*>(frame.data), dataSize);

    output_file.close();
}

void getCubicSplineInterpolation(cv::Mat A_vec, 
                            cv::Mat b_vec, 
                            cv::Mat x_vec,
                            int batchSize, 
                            int height,
                            int width)
{
   //Getting inverse of A matrix. 
   cublasStatus status; 
   cublasHandle_t handle;
   cublasCreate_v2(&handle);
  
   //Allocating memory. 
   float *d_A_vec;
   float *d_b_vec;
   float *d_x_vec;
   float alpha = 1.0f, beta = 0.0f; 


   status=cublasAlloc(A_vec.rows * A_vec.cols, sizeof(float), (void**)&d_A_vec);
   if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return;
    }

    status=cublasAlloc(b_vec.cols * b_vec.rows * width, sizeof(float), (void**)&d_b_vec);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (b)\n");
      return;
      
    }

    status=cublasAlloc(x_vec.rows * x_vec.cols , sizeof(float), (void**)&d_x_vec);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (x)\n");
      return;
    }
    
    float *A_vec_data = (float*) malloc(A_vec.rows*A_vec.cols*sizeof(float));
    float *b_vec_data = (float*) malloc(b_vec.rows*b_vec.cols*sizeof(float));
    
    memcpy(A_vec_data, A_vec.data, A_vec.rows*A_vec.cols*sizeof(float));
    memcpy(b_vec_data, b_vec.data, b_vec.rows*b_vec.cols*sizeof(float));
    
    //Copy to device
    status = cublasSetMatrix(A_vec.rows,A_vec.cols,sizeof(float),A_vec_data,A_vec.rows,d_A_vec,A_vec.rows);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory copy error (A)\n");
      return;
    }

    status=cublasSetMatrix(b_vec.rows,b_vec.cols,sizeof(float),b_vec_data,b_vec.rows,d_b_vec,b_vec.rows);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory copy error (b)\n");
      return;
    }

    cublasSgemm(
        handle,
        CUBLAS_OP_T,    // Transpose operation for A (CUBLAS_OP_N for no transpose)
        CUBLAS_OP_N,    // Transpose operation for B
        A_vec.rows,      // Number of rows of op(A) and C
        b_vec.cols,       // Number of columns of op(B) and C
        A_vec.cols,      // Number of columns of op(A) and rows of op(B)
        &alpha,      // Pointer to scalar alpha
        d_A_vec,         // Pointer to device memory of A
        A_vec.cols,           // Leading dimension of A
        d_b_vec,         // Pointer to device memory of B
        b_vec.cols,           // Leading dimension of B
        &beta,       // Pointer to scalar beta
        d_x_vec,         // Pointer to device memory of C
        x_vec.cols);          // Leading dimension of C
       
    // Copy coefficients back to the host memory
    cublasGetMatrix(x_vec.rows,x_vec.cols,sizeof(float),d_x_vec,x_vec.rows,x_vec.data,x_vec.rows);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device read error (A)\n");
      return;

    }

   free(A_vec_data);
   free(b_vec_data);
   cudaFree(d_x_vec);
   cudaFree(d_A_vec);
   cudaFree(d_b_vec);
   cublasDestroy_v2(handle);
}

int main (int argc, char* argv[]) { 
    cv::Mat cvFrame;
    int retVal=0;
    int kernelNum;

    if(argc < 3) {

        std::cerr << "Usage:" << argv[0] << " " << "<InputVideo.mp4>" << " " << "<Architecture CPU|CUDA>" << " " << "<Kernel Number 0|1|2|>" << std::endl;
        return 1;
    }
    std::string strInputVideo = argv[1];
    std::string strArch = argv[2];

    if(argv[3] != NULL){
        std::string kernel = argv[3];
        kernelNum = std::stoi(kernel);
    } else {
        // Use default kernel number
        kernelNum =1;
    }
    
    // Load the input video
    cv::VideoCapture src;
    if (!src.open(strInputVideo)) {
        throw std::runtime_error("Can't open '" + strInputVideo + "'");
    }

    // Open the output video for writing using input's characteristics
    int w = src.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = src.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    double fps = src.get(cv::CAP_PROP_FPS);
    int nframes=src.get(cv::CAP_PROP_FRAME_COUNT);

    // Create the output video
    cv::VideoWriter outVideo("denoised_" + strInputVideo, fourcc, fps, cv::Size(w, h));
    if (!outVideo.isOpened()) {
        throw std::runtime_error("Can't create output video");
    }

    //Initialize CUDA and Libraries
    if(strArch=="CUDA") {
        cudaDeviceProp devProp;
        CUDA_CHECK(cudaGetDeviceProperties(&devProp, 0)); // Get device properties
        printf("GPU Device: %s\n", devProp.name);
    }
    

    //Input and output frame. 
    cv::Mat frame; // To store the original color frame

    // Convert from RGB to Gray and Normalize frame. 
    cv::Mat prev_frame;
    cv::Mat bgr_frame;
    cv::Mat yuv_frame;
    cv::Mat out_frame;
    cv::Mat color_frame;

    bgr_frame.create(h, w, CV_8UC3);
    yuv_frame.create(h, w, CV_8UC3);
    prev_frame.create(h *3/2, w, CV_8UC1);
    out_frame.create(h * 3/2, w, CV_8UC1);
    color_frame.create(h, w, CV_8UC3);

    cv::Mat img;
    img.create(h, w, CV_8UC2);

    //set all pixels prev frame to zero
    prev_frame.setTo(cv::Scalar(0)); 
    out_frame.setTo(cv::Scalar(0)); 

    //create separate y, u, v channels
    cv::Mat y_channel;
    cv::Mat u_channel;
    cv::Mat v_channel;

    y_channel.create(h, w, CV_8UC1);
    u_channel.create(h, w, CV_8UC1);
    v_channel.create(h, w, CV_8UC1);

    cv::Mat yuv420_image;
    // Constant to run kernel
    float alpha = 0.75f;
    //int batchSize = 1;

    // Setup matrix
    cv::Mat A_vec;
    cv::Mat x_vec;
    cv::Mat b_vec;
    A_vec.create(h,w, CV_8UC3);
    b_vec.create(h,w, CV_8UC1);
    x_vec.create(h, w, CV_8UC1);
    std::vector<float> coefficients;
    std::vector<int> frames;
    float time=0;
    float totalTime=0;


    for(auto currFrame=0; currFrame<nframes; currFrame++) {
        // Preprocess Frame
        src >> frame;
        // Break the loop if no more frames are available
        if (frame.empty()) {
            break;
        }
        if(strArch == "CPU") {
             time =UsingOpenCvCPUDenoiseFunction(frame, color_frame);
             totalTime += time;
             outVideo.write(color_frame);
        }
        if(strArch == "CUDA") {

            // Convert the BGR frame to YUV
            cv::cvtColor(frame, yuv_frame, cv::COLOR_BGR2YUV);

            cv::cvtColor(frame, yuv420_image,  cv::COLOR_BGR2YUV_I420);

            // 1. Separate out Y, U, V planes.
            cv::Mat y_plane(h, w, CV_8UC1, yuv420_image.data);
            cv::Mat u_plane(h / 2, w / 2, CV_8UC1, yuv420_image.data + (w* h));
            cv::Mat v_plane(h / 2, w / 2, CV_8UC1, yuv420_image.data + (w * h) + (w * h / 4));

            // 2. Create interleaved UV plane
            cv::Mat uv_plane(h / 2, w, CV_8UC1);
            for (int i = 0; i < h / 2; ++i) {
                for (int j = 0; j < w / 2; ++j) {
                    uv_plane.at<uchar>(i, 2 * j) = u_plane.at<uchar>(i, j); // U
                    uv_plane.at<uchar>(i, 2 * j + 1) = v_plane.at<uchar>(i, j); // V
                }
            }

            // 3. Combine Y and UV planes into NV12
            cv::Mat nv12_frame(h * 3 / 2, w, CV_8UC1);
            y_plane.copyTo(nv12_frame(cv::Rect(0, 0, w, h)));
            uv_plane.copyTo(nv12_frame(cv::Rect(0, h, w, h / 2))); 


            // Call noise reduction kernel on yuv image.
            unsigned char *outFramePtr =reinterpret_cast<unsigned char*>(out_frame.data);
            unsigned char *prevFramePtr =reinterpret_cast<unsigned char*>(prev_frame.data);
            unsigned char *yuvFramePtr =reinterpret_cast<unsigned char*>(nv12_frame.data);

           
            //Apply temporal reduction on y channel with kernel number
            time = applyTemporalNoiseReduction((char*)outFramePtr,(char*)yuvFramePtr, (char*)prevFramePtr, w, h, alpha, kernelNum);
            totalTime +=time;


            //std::string filename ="prev_frame"+std::to_string(currFrame)+".bin";
            //dumpFrameToBinary(filename, prev_frame);

            cv::Mat picNV12 = cv::Mat(h * 3/2, w, CV_8UC1, nv12_frame.data);
            cv::cvtColor(picNV12, bgr_frame, cv::COLOR_YUV2BGR_NV12);

            //write video
            outVideo.write(bgr_frame);
           
            //swap the frames.
            std::swap(prevFramePtr, yuvFramePtr);

            // find x in Ax=B
            //getCubicSplineInterpolation(A_vec, b_vec, x_vec, batchSize, h, w);

            //c(t) = H3,0(t) * p0 + H3,1(t) * v0 + H3,2(t) * v1 + H3,3(t) * p1
            //Calculate single coefficient per frame. 
           // cv::Scalar meanOfSpecificColumn = cv::mean(x_vec);
           // float meanFloat0 = static_cast<float>(meanOfSpecificColumn[0]);
            //coefficients.push_back(meanFloat0);
            //frames.push_back(currFrame);

        } 

       

    }
   
   printf("Kernel execution time: %f ms\n", totalTime);   
   // Opencv frame release
   prev_frame.release();
   bgr_frame.release();
   yuv_frame.release();
   out_frame.release();

   return retVal;
}
