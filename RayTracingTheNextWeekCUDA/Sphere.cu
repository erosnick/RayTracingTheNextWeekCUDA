#include "Sphere.h"

CUDA_DEVICE bool Sphere::hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
    auto oc = ray.origin - center;
    auto a = dot(ray.direction, ray.direction);
    auto halfB = dot(oc, ray.direction);
    auto c = dot(oc, oc) - radius * radius;
    auto discriminant = halfB * halfB - a * c;
    // Cant's use Math::epsilon(0.001f) for comparison here
    // Because it's not small enough(Not precise enough)
    auto bHit = (discriminant > FLT_EPSILON);

    if (!bHit) {
        return false;
    }

    auto inversea = 1.0f / a;

    auto sqrtd = sqrt(discriminant);
    Float root = (-halfB - sqrtd) * inversea;

    // Find the nearest root that lies in the acceptable range.
    if (root < tMin || tMax < root) {
        root = (-halfB + sqrtd) * inversea;
        if (root < tMin || tMax < root) {
            return false;
        }
    }

    hitResult.t = root;
    auto position = ray.at(hitResult.t);
    auto outwardNormal = (position - center) / radius;
    hitResult.setFaceNormal(ray, outwardNormal);
    //hitResult.material = material;
    hitResult.materialId = material->id;
    return true;
}

CUDA_DEVICE bool MovingSphere::hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
    auto oc = ray.origin - center(ray.time);
    auto a = dot(ray.direction, ray.direction);
    auto halfB = dot(oc, ray.direction);
    auto c = dot(oc, oc) - radius * radius;
    auto discriminant = halfB * halfB - a * c;
    // Cant's use Math::epsilon(0.001f) for comparison here
    // Because it's not small enough(Not precise enough)
    auto bHit = (discriminant > FLT_EPSILON);

    if (!bHit) {
        return false;
    }

    auto inversea = 1.0f / a;

    auto sqrtd = sqrt(discriminant);
    Float root = (-halfB - sqrtd) * inversea;

    // Find the nearest root that lies in the acceptable range.
    if (root < tMin || tMax < root) {
        root = (-halfB + sqrtd) * inversea;
        if (root < tMin || tMax < root) {
            return false;
        }
    }

    hitResult.t = root;
    auto position = ray.at(hitResult.t);
    auto outwardNormal = (position - center(ray.time)) / radius;
    hitResult.setFaceNormal(ray, outwardNormal);
    //hitResult.material = material;
    hitResult.materialId = material->id;
    return true;
}

CUDA_DEVICE Float3 MovingSphere::center(Float time) const {
    auto newCenter = center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
    return newCenter;
}