#include "Material.h"

CUDA_DEVICE bool Lambertian::scatter(const Ray& inRay, const HitResult& hitResult, Vector3Df& attenuation, Ray& scattered, curandState* randState) const {
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
    scattered = Ray(inRay.at(hitResult.t), normalize(scatterDirection), inRay.time);
    attenuation = albedo;
    return true;
}