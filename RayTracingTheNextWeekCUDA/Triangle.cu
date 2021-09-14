#include "Triangle.h"
#include "Material.h"

CUDA_DEVICE bool Triangle::hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
    // TODO: Implement this function that tests whether the triangle
    // that's specified bt v0, v1 and v2 intersects with the ray (whose
    // origin is *orig* and direction is *dir*)
    // Also don't forget to update tnear, u and v.
    auto E1 = v1 - v0;
    auto E2 = v2 - v0;
    auto S = ray.origin - v0;
    auto S1 = cross(ray.direction, E2);
    auto S2 = cross(S, E1);
    //auto coefficient = 1.0f / dot(S1, E1);
    auto coefficient = __fdividef(1.0f, dot(S1, E1));
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
        hitResult.material = material;
        //hitResult.materialId = material->id;
        return true;
    }

    //if (dot(ray.direction, normal) > 0.0f) {
    //    return false; 
    //}

    //Float u, v, t = 0.0f;
    //Float3 pvec = cross(ray.direction, E2);
    //Float det = dot(E1, pvec);
    //if (fabs(det) < FLT_EPSILON) {
    //    return false;
    //}

    //Float detInverse = 1.0f / det;
    //Float3 tvec = ray.origin - v0;
    //u = dot(tvec, pvec) * detInverse;
    //if (u < 0.0f || u > 1.0f) {
    //    return false;
    //}
    //Float3 qvec = cross(tvec, E1);
    //v = dot(ray.direction, qvec) * detInverse;
    //if (v < 0.0f || (u + v) > 1.0f) {
    //    return false;
    //}
    //t = dot(E2, qvec) * detInverse;

    //// TODO find ray triangle intersection
    //if (t < 0.0f)
    //{
    //    return false;
    //}

    //hitResult.t = t;
    //hitResult.materialId = material->id;
    //hitResult.setFaceNormal(ray, normal);

    //return true;
    
    return false;
}