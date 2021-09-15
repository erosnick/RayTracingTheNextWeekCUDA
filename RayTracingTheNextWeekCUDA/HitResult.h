#pragma once

class Material;

#include "Ray.h"
#include "Constants.h"

struct HitResult {
    Float t = -Math::infinity;
    Vector3Df normal;
    bool bFrontFace = true;
    Material* material;
    //uint8_t materialId;

    CUDA_HOST_DEVICE inline void setFaceNormal(const Ray& ray, const Vector3Df& outwardNormal) {
        bFrontFace = dot(ray.direction, outwardNormal) < FLT_EPSILON;
        normal = bFrontFace ? outwardNormal : -outwardNormal;
    }
};