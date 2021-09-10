#pragma once

#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cudaGL.h>
#include <cuda_gl_interop.h>

class Processor {
public:
    void setInput(uint8_t* const data, int imageWidth, int imageHeight);
    void processData(uint8_t* data);
    GLuint getInputTexture();
    void writeOutputTo(uint8_t* destination);
private:
    /**
    * @brief True if the textures and surfaces are initialized.
    *
    * Prevents memory leaks
    */
    bool surfacesInitialized = false;
    /**
     * @brief The width and height of a texture/surface pair.
     *
     */
    struct ImgDim { int width, height; };
    /**
     * @brief Creates a CUDA surface object, CUDA resource, and OpenGL texture from some data.
     */
    void createTextureSurfacePair(const ImgDim& dimensions, uint8_t* const data, GLuint& textureOut, cudaGraphicsResource_t& graphicsResourceOut, cudaSurfaceObject_t& surfaceOut);
    /**
     * @brief Destroys every CUDA surface object, CUDA resource, and OpenGL texture created by this instance.
     */
    void destroyEverything();
    /**
     * @brief The dimensions of an image and its corresponding texture.
     *
     */
    ImgDim imageInputDimensions, imageOutputDimensions;
    /**
     * @brief A CUDA surface that can be read to, written from, or synchronized with a Mat or
     * OpenGL texture
     *
     */
    cudaSurfaceObject_t d_imageInputTexture = 0;
    /**
     * @brief A CUDA resource that's bound to an array in CUDA memory
     */
    cudaGraphicsResource_t d_imageInputGraphicsResource = nullptr;
    /**
     * @brief A renderable OpenGL texture that is synchronized with the CUDA data
     * @see d_imageInputTexture, d_imageOutputTexture
     */
    GLuint imageInputTexture = 0;
    /** Returns true if nothing can be rendered */
    bool empty() { return imageInputTexture == 0; }

};