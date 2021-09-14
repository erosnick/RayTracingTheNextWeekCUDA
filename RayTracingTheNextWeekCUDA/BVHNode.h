#pragma once

#include "Hitable.h" 

class BVHNode : public Hitable {
public:
    CUDA_DEVICE BVHNode() {}
    CUDA_DEVICE BVHNode() {}
    CUDA_DEVICE virtual bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const override;
    CUDA_DEVICE virtual bool boundingBox(Float time0, Float time1, AABBox& outputAABB) const override;
};