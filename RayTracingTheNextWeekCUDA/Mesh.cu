#include "Mesh.h"
#include "Triangle.h"
#include "Material.h"

CUDA_DEVICE Mesh::~Mesh() {
    for (auto i = 0; i < triangleCount; i++) {
        delete triangles[i];
    }
}

CUDA_DEVICE bool Mesh::hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
    if (!AABB.intersect(ray)) {
        return false;
    }

    //HitResult tempHitResult;
    //auto bHitAnything = false;
    //auto closestSoFar = tMax;
    //for (auto i = 0; i < triangleCount; i++) {
    //    if (triangles[i]->hit(ray, tMin, closestSoFar, tempHitResult)) {
    //        bHitAnything = true;
    //        closestSoFar = tempHitResult.t;
    //        hitResult = tempHitResult;
    //    }
    //}

    auto bHitAnything = false;
    auto closestSoFar = tMax;
    for (auto i = 0; i < triangleCount; i++) {
        // TODO: Implement this function that tests whether the triangle
        // that's specified bt v0, v1 and v2 intersects with the ray (whose
        // origin is *orig* and direction is *dir*)
        // Also don't forget to update tnear, u and v.
        auto element0 = tex1Dfetch<Float4>(triangleData, i * 3);
        auto element1 = tex1Dfetch<Float4>(triangleData, i * 3 + 1);
        auto element2 = tex1Dfetch<Float4>(triangleData, i * 3 + 2);

        Float3 v0 = { element0.x, element0.y, element0.z };
        Float3 v1 = { element1.x, element1.y, element1.z };
        Float3 v2 = { element2.x, element2.y, element2.z };

        auto E1 = v1 - v0;
        auto E2 = v2 - v0;
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
        if (((t >= FLT_EPSILON) && ((t >= tMin) && (t < closestSoFar)))
            && (b1 >= FLT_EPSILON)
            && (b2 >= FLT_EPSILON)
            && ((1.0f - b1 - b2) >= FLT_EPSILON)) {
            hitResult.t = t;
            auto normal = normalize(cross(E1, E2));
            hitResult.setFaceNormal(ray, normal);
            hitResult.materialId = material->id;
            bHitAnything = true;
            closestSoFar = t;
        }
    }

    return bHitAnything;
}