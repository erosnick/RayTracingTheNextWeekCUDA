#pragma once

#include "Hitable.h"

class Mesh : public Hitable {
public:
    CUDA_DEVICE Mesh() {}
    CUDA_DEVICE Mesh(Hitable** inTriangles, int32_t inTriangleCount, Material* inMaterial) {
        triangles = inTriangles;
        triangleCount = inTriangleCount;
        material = inMaterial;
    }

    CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const override;

    Hitable** triangles;
    int32_t triangleCount;
    Material* material;
};