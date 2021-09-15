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
    CUDA_HOST_DEVICE Plane(const Vector3Df& inPosition, const Vector3Df& inNormal, const Vector3Df& inExtend, Material* inMaterial, PlaneOrientation inOrientation = PlaneOrientation::XZ, bool bInTwoSide = true)
    : position(inPosition), normal(inNormal), extend(inExtend), material(inMaterial), orientation(inOrientation), bTwoSide(bInTwoSide) {
    }

    CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const override;
    
    CUDA_DEVICE bool boundingBox(Float time0, Float time1, AABBox& outputAABB) const override {
        return false;
    }

    Vector3Df position;
    Vector3Df normal;
    Vector3Df extend;
    PrimitiveType type = PrimitiveType::Plane;
    PlaneOrientation orientation;
    Material* material;
    bool bTwoSide;
};