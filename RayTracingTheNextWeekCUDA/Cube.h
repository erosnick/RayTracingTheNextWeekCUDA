#pragma once

#include "Hitable.h"

class Cube : public Hitable {
public:
    CUDA_DEVICE Cube() {}
    CUDA_DEVICE Cube(const Float3& inCenter, Hitable** inFaces, Material* inMaterial)
    : center(inCenter), material(inMaterial) {
        for (auto i = 0; i < 6; i++) {
            faces[i] = inFaces[i];
        }
    }

    CUDA_DEVICE ~Cube();

    CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const override;

    Float3 center;
    Hitable* faces[6];

    Material* material;
};