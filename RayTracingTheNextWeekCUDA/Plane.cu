#include "Plane.h"

CUDA_HOST_DEVICE bool Plane::hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
    // Assuming vectors are all normalized
    Float denominator = dot(normal, ray.direction);

    // The absolute value is used here, 
    // and it can be intersected from both sides 
    auto shouldProcced = bTwoSide ? (abs(denominator) > Math::epsilon) : (denominator > Math::epsilon);

    if (shouldProcced) {
        Float3 po = position - ray.origin;
        hitResult.t = dot(po, normal) / denominator;
        hitResult.position = ray.at(hitResult.t);
        hitResult.setFaceNormal(ray, normal);
        hitResult.material = material;
        auto inRange = false;
        if ((hitResult.position.x > -extend.x && hitResult.position.x < extend.x)
         && (hitResult.position.z > -extend.z && hitResult.position.z < extend.z)) {
            inRange = true;
        }
        hitResult.bHit = (hitResult.t >= tMin && hitResult.t < tMax) && inRange;
        return hitResult.bHit;
    }

    return false;
}