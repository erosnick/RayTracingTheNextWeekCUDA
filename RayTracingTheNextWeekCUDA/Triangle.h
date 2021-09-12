#pragma once

#include "Hitable.h"

class Triangle : public Hitable {
public:
    CUDA_DEVICE Triangle() {}
    CUDA_DEVICE Triangle(const Float3& inV0, const Float3& inV1, const Float3& inV2, Material* inMaterial) {
        initialize(inV0, inV1, inV2, inMaterial);
    }

    CUDA_DEVICE void initialize(const Float3& inV0, const Float3& inV1, const Float3& inV2, Material* inMaterial) {
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

    Float3 v0;
    Float3 v1;
    Float3 v2;

    Float3 E1;
    Float3 E2;

    Float3 normal;

    Material* material;

    PrimitiveType type;
};