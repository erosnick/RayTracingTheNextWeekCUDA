#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iostream>

#include "imgui/stb_image.h"
#include "imgui/stb_image_write.h"

#include "Buffer.h"
#include "Memory.h"

class Canvas {
public:
    Canvas(int32_t inWidth, int32_t inHeight) {
        initialize(inWidth, inHeight);
    }

    ~Canvas() {
        //uninitialize();
    }

    void initialize(int32_t inWidth, int32_t inHeight) {
        width = inWidth;
        height = inHeight;
        sampleCount = 0;
        renderingTime = 0.0f;

        auto size = static_cast<size_t>(width * height * 3);

        gpuErrorCheck(cudaMallocManaged(&pixelBuffer, sizeof(Buffer<uint8_t>*)));
        pixelBuffer->initialize(size);

        gpuErrorCheck(cudaMallocManaged(&accumulationBuffer, sizeof(Buffer<uint32_t>*)));
        accumulationBuffer->initialize(size);
    }

    void uninitialize() {
        accumulationBuffer->uninitialize();
        gpuErrorCheck(cudaFree(accumulationBuffer));

        pixelBuffer->uninitialize();
        gpuErrorCheck(cudaFree(pixelBuffer));
    }

    CUDA_DEVICE inline int32_t getWidth() const {
        return width;
    }

    CUDA_DEVICE inline int32_t getHeight() const {
        return height;
    }

    CUDA_DEVICE inline void writePixel(int32_t x, int32_t y, Float red, Float green, Float blue) {
        auto index = y * width + x;
        writePixel(index, red, green, blue);
    }

    CUDA_DEVICE inline void writePixel(int32_t index, Float red, Float green, Float blue) {
        Float start = 0.0f;
        Float end = 0.999f;
        (*pixelBuffer)[index * 3] = uint8_t(255.99f * clamp(sqrt(red), start, end));
        (*pixelBuffer)[index * 3 + 1] = uint8_t(255.99f * clamp(sqrt(green), start, end));
        (*pixelBuffer)[index * 3 + 2] = uint8_t(255.99f * clamp(sqrt(blue), start, end));
    }

    CUDA_DEVICE inline void writePixel(int32_t index, const Float3& color) {
        writePixel(index, color.x, color.y, color.z);
    }

    CUDA_DEVICE inline void accumulatePixel(int32_t index, const Float3& color) {
        accumulatePixel(index, color.x, color.y, color.z);
    }

    CUDA_DEVICE inline void accumulatePixel(int32_t x, int32_t y, Float red, Float green, Float blue) {
        auto index = y * width + x;
        accumulatePixel(index, red, green, blue);
    }

    CUDA_DEVICE inline void accumulatePixel(int32_t index, Float red, Float green, Float blue) {
        Float start = 0.0f;
        Float end = 0.999f;

        auto r = uint8_t(255.99f * clamp(sqrt(red), start, end));
        auto g = uint8_t(255.99f * clamp(sqrt(green), start, end));
        auto b = uint8_t(255.99f * clamp(sqrt(blue), start, end));
        (*accumulationBuffer)[index * 3] += r;
        (*accumulationBuffer)[index * 3 + 1] += g;
        (*accumulationBuffer)[index * 3 + 2] += b;
        (*pixelBuffer)[index * 3] = (*accumulationBuffer)[index * 3] / sampleCount;
        (*pixelBuffer)[index * 3 + 1] = (*accumulationBuffer)[index * 3 + 1] / sampleCount;
        (*pixelBuffer)[index * 3 + 2] = (*accumulationBuffer)[index * 3 + 2] / sampleCount;
    }

    //inline Tuple pixelAt(int32_t x, int32_t y) {
    //    return pixelData[y * width + x];
    //}

    inline uint8_t* getPixelBuffer() {
        return pixelBuffer->get();
    }

    inline std::string toPPM() const {
        auto ppm = std::string();
        ppm.append("P3\n");
        ppm.append(std::to_string(width) + " " + std::to_string(height) + "\n");
        ppm.append(std::to_string(255) + "\n");
        return ppm;
    }

    inline void writeToPPM(const std::string& path) {
        auto ppm = std::ofstream(path);

        if (!ppm.is_open()) {
            std::cout << "Open file image.ppm failed.\n";
        }

        std::stringstream ss;
        ss << toPPM();

        for (auto y = height - 1; y >= 0; y--) {
            for (auto x = 0; x < width; x++) {
                auto index = y * width + x;
                auto r = (*pixelBuffer)[index * 3];
                auto g = (*pixelBuffer)[index * 3 + 1];
                auto b = (*pixelBuffer)[index * 3 + 2];
                ss << r << ' ' << g << ' ' << b << '\n';
            }
        }

        ppm.write(ss.str().c_str(), ss.str().size());

        ppm.close();
    }

    inline void writeToPNG(const std::string& path) {
        auto* imageData = new uint8_t[width * height * 3];
        for (auto y = height - 1; y >= 0; y--) {
            for (auto x = 0; x < width; x++) {
                auto sourceIndex = y * width + x;
                auto destIndex = (height - 1 - y) * width + x;
                imageData[destIndex * 3] = (*pixelBuffer)[sourceIndex * 3];
                imageData[destIndex * 3 + 1] = (*pixelBuffer)[sourceIndex * 3 + 1];
                imageData[destIndex * 3 + 2] = (*pixelBuffer)[sourceIndex * 3 + 2];
            }
        }
        stbi_write_png("render.png", width, height, 3, imageData, width * 3);
        delete[] imageData;
    }

    CUDA_HOST_DEVICE inline void clearPixel(int32_t index, const Float3& clearColor = make_float3(0.0f, 0.0f, 0.0f)) {
        (*accumulationBuffer)[index * 3] = 0;
        (*accumulationBuffer)[index * 3 + 1] = 0;
        (*accumulationBuffer)[index * 3+ 2] = 0;
    }

    CUDA_HOST_DEVICE inline void incrementSampleCount() {
        sampleCount++;
    }

    CUDA_HOST_DEVICE inline void resetSampleCount() {
        sampleCount = 0;
    }

    CUDA_HOST_DEVICE inline void resetRenderingTime() {
        renderingTime = 0;
    }

    CUDA_HOST_DEVICE inline uint32_t getSampleCount() const {
        return sampleCount;
    }

    CUDA_HOST_DEVICE inline Float getRenderingTime() {
        return renderingTime;
    }

    CUDA_HOST_DEVICE inline void incrementRenderingTime() {
        renderingTime += 16.0f;
    }

    CUDA_HOST_DEVICE inline void print() const {
        for (auto i = 0; i < accumulationBuffer->size; i++) {
            printf("%d\n", (*accumulationBuffer)[i]);
        }
    }

private:
    Buffer<uint8_t>* pixelBuffer;
    Buffer<uint32_t>* accumulationBuffer;

    int32_t width;
    int32_t height;
    uint32_t sampleCount;

    Float renderingTime;
};

inline Canvas* createCanvas(int32_t width, int32_t height) {
    Canvas* canvas = nullptr;
    gpuErrorCheck(cudaMallocManaged(&canvas, sizeof(Canvas*)));
    canvas->initialize(width, height);
    return canvas;
}