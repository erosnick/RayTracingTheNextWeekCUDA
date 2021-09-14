#pragma once

#include "CUDATypes.h"
#include "Ray.h"

class AABBox {
public:
    CUDA_DEVICE AABBox() {}
    CUDA_DEVICE AABBox(const Float3& boundsMin, const Float3& boundsMax) {
        bounds[0] = boundsMin;
        bounds[1] = boundsMax;
    }

    CUDA_DEVICE inline bool intersect(const Ray& ray, Float tMin, Float tMax) const {
        // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
        // For instance the line equation for the x component of the bounding volume's minimum extent can be written as:
        // y = B0x
        // B0x in this example, corresponds to bounds[0].x in the code above.To find where the ray intersects this line we can write :
        // Ox + tDx = B0x (eq1) 
        // Which can solved by reordering the terms:
        // t0x = (B0x - Ox) / Dx(eq2)
        Float tMinX;
        Float tMaxX;
        Float tMinY;
        Float tMaxY;
        Float tMinZ;
        Float tMaxZ;
        tMinX = (bounds[ray.signs[0]].x - ray.origin.x) * ray.inverseDirection.x;
        tMaxX = (bounds[1 - ray.signs[0]].x - ray.origin.x) * ray.inverseDirection.x;
        tMinY = (bounds[ray.signs[1]].y - ray.origin.y) * ray.inverseDirection.y;
        tMaxY = (bounds[1 - ray.signs[1]].y - ray.origin.y) * ray.inverseDirection.y;

        if ((tMinX > tMaxY) || (tMinY > tMaxX)) {
            return false;
        }

        if (tMinY > tMinX) {
            tMinX = tMinY;
        }

        if (tMaxY < tMaxX) {
            tMaxX = tMaxY;
        }

        tMinZ = (bounds[ray.signs[2]].z - ray.origin.z) * ray.inverseDirection.z;
        tMaxZ = (bounds[1 - ray.signs[2]].z - ray.origin.z) * ray.inverseDirection.z;

        if ((tMinX > tMaxZ) || (tMinZ > tMaxX)) {
            return false;
        }

        if (tMinZ > tMinX) {
            tMinX = tMinZ;
        }

        if (tMaxZ < tMaxX) {
            tMaxX = tMaxZ;
        }

        Float t = tMinX;

        if (t < 0.0f) {
            t = tMaxX;
            if (t < 0.0f) {
                return false;
            }
        }

        return true;

        //// https://github.com/straaljager/GPU-path-tracing-with-CUDA-tutorial-2/blob/master/tutorial2_cuda_pathtracer.cu
        //// ray/box intersection
        //// for theoretical background of the algorithm see 
        //// http://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
        //// optimized code from http://www.gamedev.net/topic/495636-raybox-collision-intersection-point/
        //Float3 tMin = (bounds[0] - ray.origin) / ray.direction;
        //Float3 tMax = (bounds[1] - ray.origin) / ray.direction;

        //Float3 realMin = minf3(tMin, tMax);
        //Float3 realMax = maxf3(tMin, tMax);

        //float minMax = minf1(minf1(realMax.x, realMax.y), realMax.z);
        //float maxMin = maxf1(maxf1(realMin.x, realMin.y), realMin.z);

        //if (minMax >= maxMin) { 
        //    return maxMin > FLT_EPSILON;
        //}
        //else {
        //    return false;
        //}

        //for (int a = 0; a < 3; a++) {
        //    auto invD = 1.0f / ray.direction[a];
        //    auto t0 = (bounds[0][a] - ray.origin[a]) * invD;
        //    auto t1 = (bounds[1][a] - ray.origin[a]) * invD;

        //    if (invD < 0.0f) {
        //        Float t = t0;
        //        t0 = t1;
        //        t1 = t;
        //        //std::swap(t0, t1);
        //    }

        //    tMin = t0 > tMin ? t0 : tMin;
        //    tMax = t1 < tMax ? t1 : tMax;

        //    if (tMax <= tMin)
        //        return false;
        //}
        //return true;
    }

    Float3 bounds[2];
};

CUDA_DEVICE inline AABBox surroundingBox(const AABBox& box0, const AABBox& box1) {
    auto small = make_float3(fmin(box0.bounds[0].x, box1.bounds[0].x),
                             fmin(box0.bounds[0].y, box1.bounds[0].y),
                             fmin(box0.bounds[0].z, box1.bounds[0].z));

    auto big = make_float3(fmax(box0.bounds[1].x, box1.bounds[1].x),
                           fmax(box0.bounds[1].y, box1.bounds[1].y),
                           fmax(box0.bounds[1].z, box1.bounds[1].z));

    return AABBox(small, big);
}