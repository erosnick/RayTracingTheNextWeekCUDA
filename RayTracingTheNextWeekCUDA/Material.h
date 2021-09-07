#pragma once

#include "CUDATypes.h"
#include "Ray.h"
#include "HitResult.h"
#include "Utils.h"

class Material {
public:
    CUDA_DEVICE inline virtual bool scatter(const Ray& inRay, const HitResult& hitResult, Float3& attenuation, Ray& scattered, curandState* randState) const = 0;
};

class Lambertian : public Material {
public:
    CUDA_DEVICE Lambertian(const Float3& inAlbedo, Float inAbsorb)
        : albedo(inAlbedo), absorb(inAbsorb) {
    }

    CUDA_DEVICE inline bool scatter(const Ray& inRay, const HitResult& hitResult, Float3& attenuation, Ray& scattered, curandState* randState) const override {
        //auto scatterDirection = Utils::randomUnitVector(randState);                                         // Diffuse1
        auto scatterDirection = hitResult.normal + Utils::randomUnitVector(randState);                      // Diffuse2
        //auto scatterDirection = Utils::randomHemiSphere(hitResult.normal, randState);                       // Diffuse3
        //auto scatterDirection = hitResult.normal + Utils::randomHemiSphere(hitResult.normal, randState);    // Diffuse4
        //auto scatterDirection = hitResult.normal + Utils::randomInUnitSphere(randState);                    // Diffuse5
        // Catch degenerate scatter direction
        // If the random unit vector we generate is exactly opposite the normal vector, 
        // the two will sum to zero, which will result in a zero scatter direction vector. 
        // This leads to bad scenarios later on (infinities and NaNs),
        if (Utils::nearZero(scatterDirection)) {
            scatterDirection = hitResult.normal;
        }
        scattered = Ray(hitResult.position, normalize(scatterDirection), inRay.time);
        attenuation = albedo;
        return true;
    }
    Float3 albedo;
    Float absorb;
};

class Metal : public Material {
public:
    CUDA_DEVICE Metal(const Float3& inAlbedo, Float inFuzz = 1.0f)
    : albedo(inAlbedo), fuzz(inFuzz < 1.0f ? inFuzz : 1.0f) {
    }

    CUDA_DEVICE inline bool scatter(const Ray& inRay, const HitResult& hitResult, Float3& attenuation, Ray& scattered, curandState* randState) const override {
        auto reflected = Utils::reflect(normalize(inRay.direction), hitResult.normal);
        scattered = Ray(hitResult.position, reflected + fuzz * Utils::randomInUnitSphere(randState), inRay.time);
        attenuation = albedo;
        return (dot(scattered.direction, hitResult.normal) > 0);
    }

    Float3 albedo;
    Float fuzz;
};

class Dieletric : public Material {
public:
    CUDA_DEVICE Dieletric(Float inIndexOfRefraction)
    : indexOfRefraction(inIndexOfRefraction) {
    }

    CUDA_DEVICE bool scatter(const Ray& inRay, const HitResult& hitResult, Float3& attenuation, Ray& scattered, curandState* randState) const override {
        attenuation = make_float3(1.0f, 1.0f, 1.0f);
        auto refractionRatio = hitResult.bFrontFace ? (1.0f / indexOfRefraction) : indexOfRefraction;

        auto unitDirection = normalize(inRay.direction);
        auto cosTheta = fmin(dot(-unitDirection, hitResult.normal), 1.0f);
        auto sinTheta = sqrt(1.0f - cosTheta * cosTheta);

        bool bCannotRefract = refractionRatio * sinTheta > 1.0f;
        Float3 direction;

        // Consider Total Internal Reflection
        // One troublesome practical issue is that when the ray is in 
        // the material with the higher refractive index, there is no 
        // real solution to Snell's law, and thus there is no refraction
        // possible.If we refer back to Snell's law and the derivation of sin¦È':
        // sin¦È' = (¦Ç / ¦Ç')¡¤sin¦È
        // If the ray is inside glass and outside is air (¦Ç = 1.5 and ¦Ç¡ä = 1.0):
        // sin¦È' = (1.5 / 1.0)¡¤sin¦È
        // The value of sin¦È¡ä cannot be greater than 1. So, if,
        // (1.5 / 1.0)¡¤sin¦È > 1.0
        // the equality between the two sides of the equation is broken, and a 
        // solution cannot exist.If a solution does not exist, the glass cannot 
        // refract, and therefore must reflect the ray:
        if (bCannotRefract || reflectance(cosTheta, refractionRatio) > Utils::random(randState)) {
            direction = Utils::reflect(unitDirection, hitResult.normal);
        }
        else {
            direction = Utils::refract(unitDirection, hitResult.normal, refractionRatio);
        }

        scattered = Ray(hitResult.position, normalize(direction), inRay.time);

        return true;
    }

    Float indexOfRefraction;

private:
    CUDA_DEVICE static Float reflectance(Float cosine, Float refractionIndex) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1.0f - refractionIndex) / (1.0f + refractionIndex);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
    }
};