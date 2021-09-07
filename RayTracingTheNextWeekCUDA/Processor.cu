#include "Processor.h"

/** Macro for checking if CUDA has problems */
#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if(err != cudaSuccess) { \
      printf("Cuda error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  }

/**
 * A simple image processing kernel that copies the inverted data from the input surface to the output surface.
 */
__global__ void kernel(cudaSurfaceObject_t input, int width, int height, uint8_t* data) {

    //Get the pixel index
    unsigned int xPx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int yPx = threadIdx.y + blockIdx.y * blockDim.y;

    auto index = yPx * width + xPx;
    //Don't do any computation if this thread is outside of the surface bounds.
    if (index >= (width * height)) return;

    uchar4 pixel;
    pixel.x = data[index * 3 + 0];
    pixel.y = data[index * 3 + 1];
    pixel.z = data[index * 3 + 2];
    pixel.w = 255;
    surf2Dwrite(pixel, input, xPx * sizeof(uchar4), yPx);
}

void Processor::setInput(uint8_t* const data, int imageWidth, int imageHeight)
{
    //Same-size images don't need texture regeneration, so skip that.
    if (imageHeight == imageInputDimensions.height && imageWidth == imageInputDimensions.width) {


        /*
        Possible shortcut: we know the input is the same size as the texture and CUDA surface object.
        So instead of destroying the surface and texture, why not just overwrite them?

        That's what I try to do in the following block, but because "data" is BGR and the texture
        is RGBA, the channels get all messed up.
        */

        //Use the input surface's CUDAResourceDesc to gain access to the surface data array
#ifdef USE_1
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        cudaGetSurfaceObjectResourceDesc(&resDesc, d_imageInputTexture);
        cudaCheckError();
        uint8_t* data4 = new uint8_t[imageInputDimensions.width * imageInputDimensions.height * 4];
        for (int i = 0; i < imageInputDimensions.width * imageInputDimensions.height; i++) {
            data4[i * 4 + 0] = data[i * 3 + 0];
            data4[i * 4 + 1] = data[i * 3 + 1];
            data4[i * 4 + 2] = data[i * 3 + 2];
            data4[i * 4 + 3] = 255;
        }
        //Copy the data from the input array to the surface
//        cudaMemcpyToArray(resDesc.res.array.array, 0, 0, data, imageInputDimensions.width * imageInputDimensions.height * 3, cudaMemcpyHostToDevice);
        cudaMemcpy2DToArray(resDesc.res.array.array, 0, 0, data4, imageInputDimensions.width * 4, imageInputDimensions.width * 4, imageInputDimensions.height, cudaMemcpyHostToDevice);
        cudaCheckError();
        delete[] data4;
#endif
        //Set status flags
        surfacesInitialized = true;

        return;
    }


    //Clear everything that originally existed in the texture/surface
    destroyEverything();

    //Get the size of the image and place it here.
    imageInputDimensions.width = imageWidth;
    imageInputDimensions.height = imageHeight;
    imageOutputDimensions.width = imageWidth;
    imageOutputDimensions.height = imageHeight;

    //Create the input surface/texture pair
    createTextureSurfacePair(imageInputDimensions, data, imageInputTexture, d_imageInputGraphicsResource, d_imageInputTexture);

    //Set status flags
    surfacesInitialized = true;
}

void Processor::processData(uint8_t* data)
{
    //Call the algorithm

    //Set the number of blocks to call the kernel with.
    dim3 blockSize(32, 32);
    dim3 gridSize((imageInputDimensions.width + blockSize.x - 1) / blockSize.x,
                  (imageInputDimensions.height + blockSize.y - 1) / blockSize.y);

    kernel<<<gridSize, blockSize >>>(d_imageInputTexture, imageInputDimensions.width, imageInputDimensions.height, data);

    //Sync the surface with the texture
    cudaDeviceSynchronize();
    cudaCheckError();
}

GLuint Processor::getInputTexture()
{
    return imageInputTexture;
}

void Processor::writeOutputTo(uint8_t* destination)
{
    //Haven't figured this out yet
}

void Processor::createTextureSurfacePair(const Processor::ImgDim& dimensions, uint8_t* const data, GLuint& textureOut, cudaGraphicsResource_t& graphicsResourceOut, cudaSurfaceObject_t& surfaceOut) {

    // Create the OpenGL texture that will be displayed with GLAD and GLFW
    glGenTextures(1, &textureOut);
    // Bind to our texture handle
    glBindTexture(GL_TEXTURE_2D, textureOut);
    // Set texture interpolation methods for minification and magnification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Set texture clamping method
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    // Create the texture and its attributes
    glTexImage2D(GL_TEXTURE_2D,     // Type of texture
        0,                // Pyramid level (for mip-mapping) - 0 is the top level
        GL_RGB,          // Internal color format to convert to
        dimensions.width,            // Image width  i.e. 640 for Kinect in standard mode
        dimensions.height,           // Image height i.e. 480 for Kinect in standard mode
        0,                // Border width in pixels (can either be 1 or 0)
        GL_RGB,          // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
        GL_UNSIGNED_BYTE, // Image data type.
        data);            // The actual image data itself
    //Note that the type of this texture is an RGBA UNSIGNED_BYTE type. When CUDA surfaces
    //are synchronized with OpenGL textures, the surfaces will be of the same type.
    //They won't know or care about their data types though, for they are all just byte arrays
    //at heart. So be careful to ensure that any CUDA kernel that handles a CUDA surface
    //uses it as an appropriate type. You will see that the update_surface kernel (defined 
    //above) treats each pixel as four unsigned bytes along the X-axis: one for red, green, blue,
    //and alpha respectively.

    //Create the CUDA array and texture reference
    cudaArray* bitmap_d;
    //Register the GL texture with the CUDA graphics library. A new cudaGraphicsResource is created, and its address is placed in cudaTextureID.
    //Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL_1g80d12187ae7590807c7676697d9fe03d
    cudaGraphicsGLRegisterImage(&graphicsResourceOut, textureOut, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsNone);
    cudaCheckError();
    //Map graphics resources for access by CUDA.
    //Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1gad8fbe74d02adefb8e7efb4971ee6322
    cudaGraphicsMapResources(1, &graphicsResourceOut, 0);
    cudaCheckError();
    //Get the location of the array of pixels that was mapped by the previous function and place that address in bitmap_d
    //Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g0dd6b5f024dfdcff5c28a08ef9958031
    cudaGraphicsSubResourceGetMappedArray(&bitmap_d, graphicsResourceOut, 0, 0);
    cudaCheckError();
    //Create a CUDA resource descriptor. This is used to get and set attributes of CUDA resources.
    //This one will tell CUDA how we want the bitmap_surface to be configured.
    //Documentation for the struct: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaResourceDesc.html#structcudaResourceDesc
    struct cudaResourceDesc resDesc;
    //Clear it with 0s so that some flags aren't arbitrarily left at 1s
    memset(&resDesc, 0, sizeof(resDesc));
    //Set the resource type to be an array for convenient processing in the CUDA kernel.
    //List of resTypes: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g067b774c0e639817a00a972c8e2c203c
    resDesc.resType = cudaResourceTypeArray;
    //Bind the new descriptor with the bitmap created earlier.
    resDesc.res.array.array = bitmap_d;
    //Create a new CUDA surface ID reference.
    //This is really just an unsigned long long.
    //Docuentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gbe57cf2ccbe7f9d696f18808dd634c0a
    surfaceOut = 0;
    //Create the surface with the given description. That surface ID is placed in bitmap_surface.
    //Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__SURFACE__OBJECT.html#group__CUDART__SURFACE__OBJECT_1g958899474ab2c5f40d233b524d6c5a01
    cudaCreateSurfaceObject(&surfaceOut, &resDesc);
    cudaCheckError();
}

void Processor::destroyEverything()
{
    if (surfacesInitialized) {

        //Input image CUDA surface
        cudaDestroySurfaceObject(d_imageInputTexture);
        cudaGraphicsUnmapResources(1, &d_imageInputGraphicsResource);
        cudaGraphicsUnregisterResource(d_imageInputGraphicsResource);
        d_imageInputTexture = 0;

        //Input image GL texture
        glDeleteTextures(1, &imageInputTexture);
        imageInputTexture = 0;

        surfacesInitialized = false;
    }
}
