#include "Triangle.h"
#include "Material.h"

CUDA_DEVICE bool Triangle::hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
    // TODO: Implement this function that tests whether the triangle
    // that's specified bt v0, v1 and v2 intersects with the ray (whose
    // origin is *orig* and direction is *dir*)
    // Also don't forget to update tnear, u and v.
    auto S = ray.origin - v0;
    auto S1 = cross(ray.direction, E2);
    auto S2 = cross(S, E1);
    auto coefficient = 1.0f / dot(S1, E1);
    auto t = coefficient * dot(S2, E2);
    auto b1 = coefficient * dot(S1, S);
    auto b2 = coefficient * dot(S2, ray.direction);

    // Constrains:
    // 1 ~ 4 must be satisfied at the same time
    // 1.t must be greater than or equal to 0
    // 2.u must be a non-negative number with a 
    // value less than or equal to 1.
    // 3.v must be a non-negative number with a 
    // value less than or equal to 1.
    // 4.v + u must be a number less than or equal to 1
    if (((t >= FLT_EPSILON) && ((t >= tMin) && (t < tMax)))
        && (b1 >= FLT_EPSILON)
        && (b2 >= FLT_EPSILON)
        && ((1.0f - b1 - b2) >= FLT_EPSILON)) {
        hitResult.t = t;
        hitResult.setFaceNormal(ray, normal);
        hitResult.materialId = material->id;
        return true;
    }
    
    return false;
}