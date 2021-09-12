#pragma once

#include "Hitable.h"

class Cube : public Hitable {
public:
    CUDA_DEVICE Cube() {}
    CUDA_DEVICE Cube(const Float3& inPosition, Hitable** inFaces, Material* inMaterial)
    : position(inPosition), faces(inFaces), material(inMaterial) {
    }

    CUDA_DEVICE virtual ~Cube();

    CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const override;

    Float3 position;
    Hitable** faces;

    Material* material;
};