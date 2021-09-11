#include "Plane.h"
#include "Material.h"

CUDA_HOST_DEVICE bool Plane::hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
    // Assuming vectors are all normalized
    Float denominator = dot(normal, ray.direction);

    // The absolute value is used here, 
    // and it can be intersected from both sides 
    auto shouldProcced = bTwoSide ? (abs(denominator) > Math::epsilon) : (denominator > Math::epsilon);

    if (shouldProcced) {
        Float3 po = position - ray.origin;
        hitResult.t = dot(po, normal) / denominator;
        auto position = ray.at(hitResult.t);
        hitResult.setFaceNormal(ray, normal);
        //hitResult.material = material;
        hitResult.materialId = material->id;
        auto inRange = false;
        if ((position.x > -extend.x && position.x < extend.x)
         && (position.z > -extend.z && position.z < extend.z)) {
            inRange = true;
        }
        return (hitResult.t >= tMin && hitResult.t < tMax) && inRange;
    }

    return false;
}