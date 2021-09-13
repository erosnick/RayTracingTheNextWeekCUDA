#pragma once

#include "Hitable.h"
#include "AABBox.h"

class Mesh : public Hitable {
public:
    CUDA_DEVICE Mesh() {}
    CUDA_DEVICE Mesh(Hitable** inTriangles, int32_t inTriangleCount, cudaTextureObject_t inTriangleData, const Float3& boundsMin, const Float3& boundsMax, Material* inMaterial)
        : triangles(inTriangles), triangleCount(inTriangleCount), triangleData(inTriangleData), AABB({ boundsMin, boundsMax }), material(inMaterial) {
    }

    CUDA_DEVICE ~Mesh();

    CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const override;

    AABBox AABB;
    Hitable** triangles;
    cudaTextureObject_t triangleData;
    int32_t triangleCount;
    Material* material;
};