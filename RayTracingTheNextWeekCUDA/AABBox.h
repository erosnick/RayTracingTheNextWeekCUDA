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

    CUDA_DEVICE inline bool intersect(const Ray& ray) const {
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
    }

    Float3 bounds[2];
};