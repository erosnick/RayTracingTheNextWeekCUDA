#pragma once

#include "CUDATypes.h"
#include "HitResult.h"
#include "Material.h"
#include "Hitable.h"

class Sphere : public Hitable {
public:
    CUDA_DEVICE Sphere() {}
    CUDA_DEVICE Sphere(const Float3& inCenter, float inRadius, Material* inMaterial, bool bInShading = true) {
        initialize(inCenter, inRadius, inMaterial, bInShading);
    }

    CUDA_DEVICE void initialize(const Float3& inCenter, float inRadius, Material* inMaterial, bool bInShading = true) {
        center = inCenter;
        radius = inRadius;
        material = inMaterial;
        bShading = bInShading;
    }

    void uninitailize() {
        //if (material) {
        //    delete material;
        //}
    }

    CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const override;

    Float3 center;
    Float radius;
    Material* material;
    bool bShading;
};

class MovingSphere : public Hitable {
public:
    CUDA_DEVICE MovingSphere() {};
    CUDA_DEVICE MovingSphere(const Float3& inCenter0, const Float3& inCenter1, Float inTime0, Float inTime1, Float inRadius, Material* inMaterial) {
        initialize(inCenter0, inCenter1, inTime0, inTime1, inRadius, inMaterial);

    }

    CUDA_DEVICE void initialize(const Float3& inCenter0, const Float3& inCenter1, Float inTime0, Float inTime1, Float inRadius, Material* inMaterial) {
        center0 = inCenter0;
        center1 = inCenter1;
        time0 = inTime0;
        time1 = inTime1;
        radius = inRadius;
        material = inMaterial;
    }

    CUDA_DEVICE Float3 center(Float time) const;

    CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const override;

    Float3 center0;
    Float3 center1;
    Float time0;
    Float time1;
    Float radius;
    Material* material;
};