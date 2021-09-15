#pragma once

#include "Hitable.h"
#include "AABBox.h"

struct TriangleMeshData {
    int32_t triangleCount;
    Vector3Df boundsMin;
    Vector3Df boundsMax;
    Float4* vertices;
    Float4* AABBs;
    cudaTextureObject_t triangleData;
    cudaTextureObject_t AABBData;
};

class TriangleMesh : public Hitable {
public:
    CUDA_DEVICE TriangleMesh() {}
    CUDA_DEVICE TriangleMesh(int32_t inTriangleCount, cudaTextureObject_t inTriangleData, cudaTextureObject_t inAABBData, const Vector3Df& boundsMin, const Vector3Df& boundsMax, Material* inMaterial)
    : triangleCount(inTriangleCount), triangleData(inTriangleData), AABBData(inAABBData), AABB({ boundsMin, boundsMax }), material(inMaterial) {
    }

    CUDA_DEVICE ~TriangleMesh();

    CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const override;

    CUDA_DEVICE bool boundingBox(Float time0, Float time1, AABBox& outputAABB) const override {
        outputAABB = AABB;
        return true;
    }

    AABBox AABB;
    cudaTextureObject_t triangleData;
    cudaTextureObject_t AABBData;
    int32_t triangleCount;
    Material* material;
};