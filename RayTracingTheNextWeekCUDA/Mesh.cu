#include "Mesh.h"
#include "Triangle.h"

CUDA_DEVICE bool Mesh::hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
    HitResult tempHitResult;
    auto bHitAnything = false;
    auto closestSoFar = tMax;
    for (auto i = 0; i < triangleCount; i++) {
        if (triangles[i]->hit(ray, tMin, closestSoFar, tempHitResult)) {
            bHitAnything = true;
            closestSoFar = tempHitResult.t;
            hitResult = tempHitResult;
        }
    }

    return bHitAnything;
}