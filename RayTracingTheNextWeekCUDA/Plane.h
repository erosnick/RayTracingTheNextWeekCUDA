#pragma once

#include "Hitable.h"

enum class PlaneOrientation : uint8_t {
    XY,
    YZ,
    XZ
};

class Plane : public Hitable {
public:
    CUDA_HOST_DEVICE Plane() {}
    CUDA_HOST_DEVICE Plane(const Float3& inPosition, const Float3& inNormal, const Float3& inExtend, Material* inMaterial, bool bInTwoSide = false)
    : position(inPosition), normal(inNormal), extend(inExtend), material(inMaterial), bTwoSide(bInTwoSide) {}

    CUDA_HOST_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const override;

    Float3 position;
    Float3 normal;
    Float3 extend;
    PrimitiveType type = PrimitiveType::Plane;
    Material* material;
    bool bTwoSide;
};