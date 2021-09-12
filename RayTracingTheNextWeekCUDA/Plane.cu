#include "Plane.h"
#include "Material.h"

CUDA_DEVICE bool Plane::hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
    // Assuming vectors are all normalized
    Float denominator = dot(normal, ray.direction);

    // The absolute value is used here, 
    // and it can be intersected from both sides 
    auto shouldProcced = bTwoSide ? (abs(denominator) > Math::epsilon) : (denominator > Math::epsilon);

    if (shouldProcced) {
        Float3 po = position - ray.origin;
        hitResult.t = dot(po, normal) / denominator;
        auto hitPosition = ray.at(hitResult.t);
        hitResult.setFaceNormal(ray, normal);
        //hitResult.material = material;
        hitResult.materialId = material->id;
        auto inRange = false;

        auto leftBoundary = position - extend;
        auto rightBoundary = position + extend;

        switch (orientation)
        {
        case PlaneOrientation::XY:
            if (((hitPosition.x > leftBoundary.x) && (hitPosition.x < rightBoundary.x))
             && ((hitPosition.y > leftBoundary.y) && (hitPosition.y < rightBoundary.y))) {
                inRange = true;
            }
            break;
        case PlaneOrientation::YZ:
            if (((hitPosition.y > position.y - extend.y) && (hitPosition.y < position.y + extend.y))
             && ((hitPosition.z > position.z - extend.z) && (hitPosition.z < position.z + extend.z))) {
                inRange = true;
            }
            break;
        case PlaneOrientation::XZ:
            if (((hitPosition.x > leftBoundary.x) && (hitPosition.x < rightBoundary.x))
             && ((hitPosition.z > leftBoundary.z) && (hitPosition.z < rightBoundary.z))) {
                inRange = true;
            }
            break;
        default:
            break;
        }

        return ((hitResult.t >= tMin) && (hitResult.t < tMax)) && inRange;
    }

    return false;
}