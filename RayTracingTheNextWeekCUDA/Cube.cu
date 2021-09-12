#include "Cube.h"

CUDA_DEVICE Cube::~Cube() {
    for (auto i = 0; i < 6; i++) {
        delete faces[i];
    }
}

CUDA_DEVICE bool Cube::hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
    HitResult tempHitResult;
    auto bHitAnything = false;
    auto closestSoFar = tMax;
    for (auto i = 0; i < 6; i++) {
        if (faces[i]->hit(ray, tMin, closestSoFar, tempHitResult)) {
            bHitAnything = true;
            closestSoFar = tempHitResult.t;
            hitResult = tempHitResult;
        }
    }

    return bHitAnything;
}