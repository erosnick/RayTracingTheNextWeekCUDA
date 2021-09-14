#pragma once

#include "Hitable.h"
#include "AABBox.h"

struct TriangleMeshData {
    int32_t triangleCount;
    Float3 boundsMin;
    Float3 boundsMax;
    Float4* data;
    cudaTextureObject_t texture;
};

class TriangleMesh : public Hitable {
public:
    CUDA_DEVICE TriangleMesh() {}
    CUDA_DEVICE TriangleMesh(int32_t inTriangleCount, cudaTextureObject_t inTriangleData, const Float3& boundsMin, const Float3& boundsMax, Material* inMaterial)
    : triangleCount(inTriangleCount), triangleData(inTriangleData), AABB({ boundsMin, boundsMax }), material(inMaterial) {
    }

    CUDA_DEVICE ~TriangleMesh();

    CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const override;

    CUDA_DEVICE bool boundingBox(Float time0, Float time1, AABBox& outputAABB) const override {
        outputAABB = AABB;
        return true;
    }

    AABBox AABB;
    cudaTextureObject_t triangleData;
    int32_t triangleCount;
    Material* material;
};