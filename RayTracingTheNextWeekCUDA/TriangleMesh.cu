#include "TriangleMesh.h"
#include "Triangle.h"
#include "Material.h"

CUDA_DEVICE TriangleMesh::~TriangleMesh() {
}

__device__ float rayTriangleIntersection(const Ray& ray,
    const float3& v0,
    const float3& edge1,
    const float3& edge2)
{

    float3 tvec = ray.origin - v0;
    float3 pvec = cross(ray.direction, edge2);
    float  det = dot(edge1, pvec);

    det = __fdividef(1.0f, det);  // CUDA intrinsic function 

    float u = dot(tvec, pvec) * det;

    if (u < 0.0f || u > 1.0f)
        return -1.0f;

    float3 qvec = cross(tvec, edge1);

    float v = dot(ray.direction, qvec) * det;

    if (v < 0.0f || (u + v) > 1.0f)
        return -1.0f;

    return dot(edge2, qvec) * det;
}

CUDA_DEVICE bool TriangleMesh::hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
    if (!AABB.intersect(ray, tMin, tMax)) {
        return false;
    }

    auto bHitAnything = false;
    auto closestSoFar = tMax;
    for (auto i = 0; i < triangleCount; i++) {
        //// TODO: Implement this function that tests whether the triangle
        //// that's specified bt v0, v1 and v2 intersects with the ray (whose
        //// origin is *orig* and direction is *dir*)
        //// Also don't forget to update tnear, u and v.
        auto element0 = tex1Dfetch<Float4>(triangleData, i * 3);
        auto element1 = tex1Dfetch<Float4>(triangleData, i * 3 + 1);
        auto element2 = tex1Dfetch<Float4>(triangleData, i * 3 + 2);

        Float3 v0 = { element0.x, element0.y, element0.z };
        Float3 E1 = { element1.x, element1.y, element1.z };
        Float3 E2 = { element2.x, element2.y, element2.z };

        auto normal = normalize(cross(E1, E2));

        // Backface cull
        if (dot(ray.direction, normal) > FLT_EPSILON) {
            continue;
        }

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
        if (((t >= FLT_EPSILON) && ((t >= tMin) && (t < closestSoFar)))
            && (b1 >= FLT_EPSILON)
            && (b2 >= FLT_EPSILON)
            && ((1.0f - b1 - b2) >= FLT_EPSILON)) {
            hitResult.t = t;
            hitResult.setFaceNormal(ray, normal);
            //hitResult.materialId = material->id;
            hitResult.material = material;
            bHitAnything = true;
            closestSoFar = t;
        }

        //float4 v0 = tex1Dfetch<float4>(triangleData, i * 3);
        //float4 edge1 = tex1Dfetch<float4>(triangleData, i * 3 + 1);
        //float4 edge2 = tex1Dfetch<float4>(triangleData, i * 3 + 2);

        //auto normal = normalize(cross(make_float3(edge1.x, edge1.y, edge1.z), 
        //                              make_float3(edge2.x, edge2.y, edge2.z)));

        //// Backface cull
        //if (dot(ray.direction, normal) > FLT_EPSILON) {
        //    continue;
        //}

        //auto t = rayTriangleIntersection(ray,
        //    make_float3(v0.x, v0.y, v0.z),
        //    make_float3(edge1.x, edge1.y, edge1.z),
        //    make_float3(edge2.x, edge2.y, edge2.z));

        //if (((t >= tMin) && (t < closestSoFar))) {
        //    hitResult.t = t;
        //    hitResult.setFaceNormal(ray, normal);
        //    //hitResult.materialId = material->id;
        //    hitResult.material = material;
        //    bHitAnything = true;
        //    closestSoFar = t;
        //}
    }

    return bHitAnything;
}