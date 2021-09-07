#pragma once

#include <cstdint>
#include <memory>

#include "Camera.h"
#include "Canvas.h"

struct ImageData {
    ImageData()
        : data(nullptr) {}
    ~ImageData() {
    }

    uint8_t* data = nullptr;
    int32_t size = 0;
    int32_t width = 0;
    int32_t height = 0;
    int32_t channels = 0;
};

extern std::shared_ptr<ImageData> imageData;

extern Camera* camera;
extern Canvas* canvas;

extern int32_t width;
extern int32_t height;

void initialize(int32_t width, int32_t height);
void pathTracing();
void cleanup();