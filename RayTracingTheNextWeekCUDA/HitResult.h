#pragma once

class Material;

#include "Ray.h"
#include "Constants.h"

struct HitResult {
    bool bHit = false;
    Float t = -Math::infinity;
    Float3 position;
    Float3 normal;
    bool bFrontFace = true;
    Material* material;

    CUDA_HOST_DEVICE inline void setFaceNormal(const Ray& ray, const Float3& outwardNormal) {
        bFrontFace = dot(ray.direction, outwardNormal) < Math::epsilon;
        normal = bFrontFace ? outwardNormal : -outwardNormal;
    }
};