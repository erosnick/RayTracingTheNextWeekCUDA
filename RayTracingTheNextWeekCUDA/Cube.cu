#include "Cube.h"

CUDA_DEVICE Cube::~Cube() {
    for (auto face : faces) {
        delete face;
    }
    printf("Cube::~Cube()\n");
}

CUDA_DEVICE bool Cube::hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
    HitResult tempHitResult;
    auto bHitAnything = false;
    auto closestSoFar = tMax;
    for (auto face : faces) {
        if (face->hit(ray, tMin, closestSoFar, tempHitResult)) {
            bHitAnything = true;
            closestSoFar = tempHitResult.t;
            hitResult = tempHitResult;
        }
    }

    return bHitAnything;
}