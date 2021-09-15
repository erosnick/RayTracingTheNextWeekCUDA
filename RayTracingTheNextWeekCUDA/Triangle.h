#pragma once

#include "Hitable.h"

class Triangle : public Hitable {
public:
    CUDA_DEVICE Triangle() {}
    CUDA_DEVICE Triangle(const Vector3Df& inV0, const Vector3Df& inV1, const Vector3Df& inV2, Material* inMaterial) {
        initialize(inV0, inV1, inV2, inMaterial);
    }

    CUDA_DEVICE void initialize(const Vector3Df& inV0, const Vector3Df& inV1, const Vector3Df& inV2, Material* inMaterial) {
        v0 = inV0;
        v1 = inV1;
        v2 = inV2;

        E1 = v1 - v0;
        E2 = v2 - v0;

        normal = normalize(cross(E1, E2));

        material = inMaterial;

        type = PrimitiveType::Triangle;
    }

    CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const override;

    CUDA_DEVICE bool boundingBox(Float time0, Float time1, AABBox& outputAABB) const override {
        return false;
    }

    Vector3Df v0;
    Vector3Df v1;
    Vector3Df v2;

    Vector3Df E1;
    Vector3Df E2;

    Vector3Df normal;

    Material* material;

    PrimitiveType type;
};