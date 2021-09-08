
#include "main.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include "Utils.h"
#include "GPUTimer.h"
#include "Sphere.h"
#include "Plane.h"
#include <yaml-cpp/yaml.h>

#include <cstdio>

template<typename T>
T* createObjectPtr() {
    T* object = nullptr;
    gpuErrorCheck(cudaMallocManaged(&object, sizeof(T*)));
    return object;
}

template<typename T>
T* createObjectArray(int32_t numObjects) {
    T* object = nullptr;
    gpuErrorCheck(cudaMallocManaged(&object, sizeof(T) * numObjects));
    return object;
}

template<typename T>
T* createObjectPtrArray(int32_t numObjects) {
    T* object = nullptr;
    gpuErrorCheck(cudaMallocManaged(&object, sizeof(T) * numObjects));
    return object;
}

template<typename T>
void deleteObject(T* object) {
    gpuErrorCheck(cudaFree(object));
}

template<typename T>
CUDA_GLOBAL void deleteDeviceObject(T** object) {
    delete (*object);
}

template<typename T>
CUDA_GLOBAL void deleteDeviceObjectArray(T** object, int32_t count) {
    for (auto i = 0; i < count; i++) {
        delete *(object + i);
    }
}

constexpr auto SPHERES = 5;
constexpr auto BOUNCES = 4;
CUDA_CONSTANT Hitable* constantSpheres[SPHERES];

CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult, Hitable** spheres) {
    HitResult tempHitResult;
    bool bHitAnything = false;
    Float closestSoFar = tMax;
    for (auto& sphere : constantSpheres) {
    //for (auto i = 0; i < SPHERES; i++){
    //    auto sphere = spheres[i];
        if (sphere->hit(ray, tMin, closestSoFar, tempHitResult)) {
            bHitAnything = true;
            closestSoFar = tempHitResult.t;
            hitResult = tempHitResult;
        }
    }

    return bHitAnything;
}

CUDA_DEVICE Float3 rayColor(const Ray& ray, curandState* randState, Hitable** spheres) {
    Ray currentRay = ray;
    auto currentAttenuation = make_float3(1.0f, 1.0f, 1.0f);
    for (auto i = 0; i < BOUNCES; i++) {
        HitResult hitResult;
        // Smaller tMin will has a impact on performance
        if (hit(currentRay, Math::epsilon, Math::infinity, hitResult, spheres)) {
            Float3 attenuation;
            Ray scattered;
            if (hitResult.material->scatter(currentRay, hitResult, attenuation, scattered, randState)) {
                currentAttenuation *= attenuation;
                currentRay = scattered;
            }
            else {
                return currentAttenuation * attenuation;
            }
        }
        // If no intersection in the first bounce, just return background color
        // otherwise return currentAttenuation * background color
        else {
            auto unitDirection = normalize(currentRay.direction);
            auto t = 0.5f * (unitDirection.y + 1.0f);
            auto background = lerp(make_float3(1.0f, 1.0f, 1.0f), make_float3(0.5f, 0.7f, 1.0f), t);
            return currentAttenuation * background;
        }
    }

    // exceeded recursion
    return make_float3(0.0f, 0.0f, 0.0f);
}

//CUDA_DEVICE Float3 rayColor(const Ray& ray, curandState* randState, Sphere* spheres, int32_t depth) {
//    if (depth == 0) {
//        // exceeded recursion
//        return make_float3(0.0f, 0.0f, 0.0f);
//    }
//    HitResult hitResult;
//    // Smaller tMin will has a impact on performance
//    if (hit(ray, Math::epsilon, Math::infinity, hitResult, spheres)) {
//        Float3 attenuation;
//        Ray rayScattered;
//        if (hitResult.material->scatter(ray, hitResult, attenuation, rayScattered, randState)) {
//            return attenuation * rayColor(rayScattered, randState, spheres, depth - 1);
//        }
//        else {
//            return make_float3(0.0f, 0.0f, 0.0f);
//        }
//    }
//
//    auto unitDirection = normalize(ray.direction);
//    auto t = 0.5f * (unitDirection.y + 1.0f);
//    auto background = lerp(make_float3(1.0f, 1.0f, 1.0f), make_float3(0.5f, 0.7f, 1.0f), t);
//    return background;
//}

CUDA_GLOBAL void renderInit(int32_t width, int32_t height, curandState* randState) {
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;
    auto index = y * width + x;

    if (index < (width * height)) {
        //Each thread gets same seed, a different sequence number, no offset
        curand_init(1984, index, 0, &randState[index]);
    }
}

//CUDA_GLOBAL void render(Canvas canvas, Camera camera, curandState* randStates, Sphere* spheres) {
//    auto x = threadIdx.x + blockDim.x * blockIdx.x;
//    auto y = threadIdx.y + blockDim.y * blockIdx.y;
//    auto width = canvas.getWidth();
//    auto height = canvas.getHeight();
//    constexpr auto samplesPerPixel = 1;
//    constexpr auto maxDepth = 5;
//    auto index = y * width + x;
//
//    if (index < (width * height)) {
//        auto color = make_float3(0.0f, 0.0f, 0.0f);
//        auto localRandState = randStates[index];
//        for (auto i = 0; i < samplesPerPixel; i++) {
//
//            auto rx = curand_uniform(&localRandState);
//            auto ry = curand_uniform(&localRandState);
//
//            auto dx = Float(x + rx) / (width - 1);
//            auto dy = Float(y + ry) / (height - 1);
//
//            auto ray = camera.getRay(dx, dy);
//            color += rayColor(ray, &localRandState, spheres);
//        }
//        // Very important!!!
//        randStates[index] = localRandState;
//        canvas.writePixel(index, color / samplesPerPixel);
//    }
//}

CUDA_GLOBAL void renderKernel(Canvas* canvas, Camera* camera, curandState* randStates, Hitable** spheres, int32_t* counter) {
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;
    auto width = canvas->getWidth();
    auto height = canvas->getHeight();
#ifdef GPU_REALTIME
    constexpr auto samplesPerPixel = 1;
#else
    constexpr auto samplesPerPixel = 1;
#endif // GPU_REALTIME

    constexpr auto maxDepth = 5;
    auto index = y * width + x;

    if (index < (width * height)) {
        auto color = make_float3(0.0f, 0.0f, 0.0f);
        auto localRandState = randStates[index];
        for (auto i = 0; i < samplesPerPixel; i++) {

            auto rx = curand_uniform(&localRandState);
            auto ry = curand_uniform(&localRandState);

            auto dx = Float(x + rx) / (width - 1);
            auto dy = Float(y + ry) / (height - 1);

            auto ray = camera->getRay(dx, dy, &localRandState);
            color += rayColor(ray, &localRandState, spheres);
        }
        // Very important!!!
        randStates[index] = localRandState; 
#ifdef GPU_REALTIME
        canvas->accumulatePixel(index, color);
#else
        //canvas->writePixel(index, color / samplesPerPixel);
        canvas->incrementSampleCount();
        canvas->accumulatePixel(index, color);

        auto tenPercent = (width * height) / 10;

        auto old = atomicAdd(counter, 1);

        if ((old + 1) > 0 && (old + 1) % tenPercent == 0) {
            printf("%.2f%%\n", (float((old + 1) * 100) / (width * height)));
        }
#endif // GPU_REALTIME
    }
}

CUDA_GLOBAL void createLambertianMaterialKernel(Material** material, Float3 albedo, Float absorb = 1.0f) {
    (*material) = new Lambertian(albedo, absorb);
}

void createLambertianMaterial(Material** material, Float3 albedo, Float absorb = 1.0f) {
    createLambertianMaterialKernel<<<1, 1>>>(material, albedo, absorb);
    gpuErrorCheck(cudaDeviceSynchronize());
}

CUDA_GLOBAL void createEmissionMaterialKernel(Material** material, Float3 albedo, Float intensity = 1.0f) {
    (*material) = new Emission(albedo, intensity);
}

void createEmissionMaterial(Material** material, Float3 albedo, Float intensity = 1.0f) {
    createEmissionMaterialKernel <<<1, 1>>>(material, albedo, intensity);
    gpuErrorCheck(cudaDeviceSynchronize());
}

CUDA_GLOBAL void createMetalMaterialKernel(Material** material, Float3 albedo, Float fuzz = 1.0f) {
    (*material) = new Metal(albedo, fuzz);
}

void createMetalMaterial(Material** material, Float3 albedo, Float fuzz = 1.0f) {
    createMetalMaterialKernel<<<1, 1>>>(material, albedo, fuzz);
    gpuErrorCheck(cudaDeviceSynchronize());
}

CUDA_GLOBAL void createDieletricMaterialKernel(Material** material, Float indexOfRefraction = 1.5f) {
    (*material) = new Dieletric(indexOfRefraction);
}

void createDieletricMaterial(Material** material, Float indexOfRefraction = 1.5f) {
    createDieletricMaterialKernel<<<1, 1>>>(material, indexOfRefraction);
    gpuErrorCheck(cudaDeviceSynchronize());
}

CUDA_GLOBAL void clearBackBuffers(Canvas* canvas) {
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;
    auto width = canvas->getWidth();
    auto height = canvas->getHeight();

    auto index = y * width + x;

    if (index < (width * height)) {
        canvas->clearPixel(index);
    }
}

CUDA_GLOBAL void createSphereKernel(Hitable** sphere, int32_t index, Float3 center, Float radius, Material* material, bool bShading) {
    *(sphere + index) = new Sphere(center, radius, material, bShading);
}

void createSphere(Hitable** sphere, int32_t index, Float3 center, Float radius, Material* material, bool bShading = true) {
    createSphereKernel<<<1, 1>>>(sphere, index, center, radius, material, bShading);
    gpuErrorCheck(cudaDeviceSynchronize());
}

CUDA_GLOBAL void createMovingSphereKernel(Hitable** sphere, int32_t index, Float3 center0, Float3 center1, Float time0, Float time1, Float radius, Material* material) {
    *(sphere + index) = new MovingSphere(center0, center1, time0, time1, radius, material);
}

void createMovingSphere(Hitable** sphere, int32_t index, Float3 center0, Float3 center1, Float time0, Float time1, Float radius, Material* material) {
    createMovingSphereKernel<<<1, 1>>>(sphere, index, center0, center1, time0, time1, radius, material);
    gpuErrorCheck(cudaDeviceSynchronize());
}

CUDA_GLOBAL void createPlaneKernel(Hitable** plane , int32_t index, Float3 position, Float3 normal, Float3 extend, Material* material) {
    *(plane + index) = new Plane(position, normal, extend, material);
}

void createPlane(Hitable** plane, int32_t index, const Float3& position, const Float3& normal, const Float3& extend, Material* material) {
    createPlaneKernel<<<1, 1>>>(plane, index, position, normal, extend, material);
    gpuErrorCheck(cudaDeviceSynchronize());
}

#define RESOLUTION 1

#if RESOLUTION == 0
int32_t width = 512;
int32_t height = 384;
#elif RESOLUTION == 1
int32_t width = 512;
int32_t height = 512;
#elif RESOLUTION == 2
int32_t width = 1024;
int32_t height = 576;
#elif RESOLUTION == 3
int32_t width = 1280;
int32_t height = 720;
#elif RESOLUTION == 4
int32_t width = 1920;
int32_t height = 1080;
#elif RESOLUTION == 5
int32_t width = 64;
int32_t height = 36;
#endif

#define SCENE 0

int32_t sampleCount = 0;

Canvas* canvas = nullptr;
Camera* camera = nullptr;
Hitable** spheres = nullptr;
constexpr auto MATERIALS = 10;
//Material** materials[MATERIALS];
std::vector<Material**> materials(MATERIALS);
curandState* randStates = nullptr;
std::shared_ptr<ImageData> imageData = nullptr;

dim3 blockSize(32, 32);
dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
              (height + blockSize.y - 1) / blockSize.y);

void loadScene(const std::string& path) {
    YAML::Node config = YAML::LoadFile(path);

    printf("name:%s\n", config["name"].as<std::string>().c_str());
    printf("sex:%s\n", config["sex"].as<std::string>().c_str());
    printf("age:%d\n", config["age"].as<int>());

    for (auto iterator = config["skills"].begin(); iterator != config["skills"].end(); iterator++) {
        printf("%s\n", iterator->first.as<std::string>().c_str());
    }
}

namespace YAML {
    template<>
    struct convert<Float3> {
        static Node encode(const Float3& rhs) {
            Node node;
            node.push_back(rhs.x);
            node.push_back(rhs.y);
            node.push_back(rhs.z);
            return node;
        }

        static bool decode(const Node& node, Float3& rhs) {
            if (!node.IsSequence() || node.size() != 3) {
                return false;
            }

            rhs.x = node[0].as<Float>();
            rhs.y = node[1].as<Float>();
            rhs.z = node[2].as<Float>();
            return true;
        }
    };

    template<>
    struct convert<Sphere> {
        static Node encode(const Sphere& rhs) {
            Node node;
            node["center"].push_back(rhs.center.x);
            node["center"].push_back(rhs.center.y);
            node["center"].push_back(rhs.center.y);
            node["radius"] = rhs.radius;
            node["type"] = (uint32_t)PrimitiveType::Sphere;

            return node;
        }

        static bool decode(const Node& node, Sphere& rhs) {
            //if (!node.IsMap() || node.size() != 3) {
            //    return false;
            //}
            printf("%d\n", node.size());

            rhs.center.x = node["center"][0].as<Float>();
            rhs.center.y = node["center"][1].as<Float>();
            rhs.center.z = node["center"][2].as<Float>();
            rhs.radius = node["radius"].as<Float>();
            rhs.primitiveType = static_cast<PrimitiveType>(node["type"].as<uint32_t>());

            return true;
        }
    };
}

void initialize(int32_t width, int32_t height) {
    //Canvas canvas(width, height);
    Utils::reportGPUUsageInfo();
    canvas = createObjectPtr<Canvas>();
    canvas->initialize(width, height);

    //Camera camera(make_float3(-2.0f, 2.0f, 1.0f), make_float3(0.0f, 0.0f, -1.0f), make_float3(0.0f, 1.0f, 0.0f), Float(width) / height, 20.0f);
    camera = createObjectPtr<Camera>();
    //camera->initialize(make_float3(-2.0f, 2.0f, 1.0f), make_float3(0.0f, 0.0f, -1.0f), make_float3(0.0f, 1.0f, 0.0f), Float(width) / height, 20.0f);
    //camera->initialize(make_float3(0.0f, 1.0f, 1.0f), make_float3(0.0f, 0.0f, -1.0f), make_float3(0.0f, 1.0f, 0.0f), Float(width) / height, 90.0f);

    //auto eye = make_float3(3.0f, 3.0f, 5.0f);
    //auto center = make_float3(0.0f, 0.0f, -1.0f);
    //auto up = make_float3(0.0f, 1.0f, 0.0f);
    //auto focusDistance = length(center - eye);
    //camera->initialize(eye, center, up, Float(width) / height, 20.0f, 2.0f, focusDistance);

    // If the distance between object and camera equals to focus lens
    // then the object is in focus
    //auto eye = position(3.0f, 3.0f, 5.0f);
    //auto center = position(0.0f, 0.0f, -1.0f);
    //auto up = position(0.0f, 1.0f, 0.0f);
    //auto focusDistance = length(center - eye);
    //camera->initialize(eye, center, up, Float(width) / height, 20.0f, 2.0f, focusDistance);

    //loadScene("./resources/scenes/test.yaml");

    for (auto& material : materials) {
        material = createObjectPtr<Material*>();
    }

    spheres = createObjectPtrArray<Hitable*>(SPHERES);

#if SCENE == 0
    // If the distance between object and camera equals to focus lens
    // then the object is in focus
    auto eye = point(3.0f, 3.0f, 5.0f);
    auto center = point(0.0f, 0.0f, -1.0f);
    auto up = point(0.0f, 1.0f, 0.0f);
    auto focusDistance = length(center - eye);
    auto aperture = 0.0f;
    camera->initialize(eye, center, up, Float(width) / height, 20.0f, aperture, focusDistance, 0.0f, 1.0f);

    // Scene1 Defocus Blur
    createDieletricMaterial(materials[0], 1.5f);
    createDieletricMaterial(materials[1], 1.5f);
    createLambertianMaterial(materials[2], make_float3(0.1f, 0.2f, 0.5f));
    //createDieletricMaterial<<<1, 1>>>(materials[3], 1.5f);
    createMetalMaterial(materials[3], make_float3(0.8f, 0.6f, 0.2f), 0.0f);
    //createLambertianMaterial<<<1, 1>>>(materials[4], make_float3(0.8f, 0.8f, 0.0f));
    createMetalMaterial(materials[4], make_float3(0.5f, 0.7f, 1.0f), 0.0f);

    auto center1 = point(0.0f, 0.5f, 0.0f);

    //createSphere(spheres, 0, { -1.0f, 0.0f, -1.0f}, 0.5f, *(materials[0]));
    //createSphere(spheres, 1, { -1.0f, 0.0f, -1.0f }, -0.4f, *(materials[1]), false);
    //createSphere(spheres, 2, {  0.0f, 0.0f, -1.0f },  0.5f, *(materials[2]));
    //createSphere(spheres, 3, {  1.0f, 0.0f, -1.0f },  0.5f, *(materials[3]));
    //createSphere(spheres, 4, {  0.0f, -100.5f, -1.0f }, 100.0f, *(materials[4]));
    createSphere(spheres, 0, { -1.0f, 0.0f, -1.0f}, 0.5f, *(materials[0]));
    createSphere(spheres, 1, { -1.0f, 0.0f, -1.0f }, -0.4f, *(materials[1]), false);
    createMovingSphere(spheres, 2, {  0.0f, 0.0f, -1.0f }, { 0.0f, 0.5f, -1.0f }, 0.0f, 1.0f, 0.5f, *(materials[2]));
    createSphere(spheres, 3, {  1.0f, 0.0f, -1.0f },  0.5f, *(materials[3]));
    createSphere(spheres, 4, {  0.0f, -100.5f, -1.0f }, 100.0f, *(materials[4]));
#elif SCENE == 1
    // If the distance between object and camera equals to focus lens
// then the object is in focus
    auto eye = point(0.0f, 0.0f, 1.0f);
    auto center = point(0.0f, 0.0f, -1.0f);
    auto up = point(0.0f, 1.0f, 0.0f);
    auto focusDistance = length(center - eye);
    auto aperture = 0.0f;
    camera->initialize(eye, center, up, Float(width) / height, 60.0f, aperture, focusDistance, 0.0f, 1.0f);

    // Scene1 Defocus Blur
    createDieletricMaterial(materials[0], 1.5f);
    createDieletricMaterial(materials[1], 1.5f);
    createLambertianMaterial(materials[2], make_float3(1.0f, 1.0f, 1.0f));
    //createDieletricMaterial<<<1, 1>>>(materials[3], 1.5f);
    createMetalMaterial(materials[3], make_float3(0.8f, 0.6f, 0.2f), 0.0f);
    //createLambertianMaterial<<<1, 1>>>(materials[4], make_float3(0.8f, 0.8f, 0.0f));
    createMetalMaterial(materials[4], make_float3(0.5f, 0.7f, 1.0f), 0.0f);

    createLambertianMaterial(materials[5], make_float3(1.0f, 1.0f, 1.0f));
    createLambertianMaterial(materials[6], make_float3(1.0f, 0.0f, 0.0f));
    createLambertianMaterial(materials[7], make_float3(0.0f, 1.0f, 0.0f));
    createEmissionMaterial(materials[8], make_float3(1.0f, 1.0f, 1.0f), 10.0f);

    auto center1 = point(0.0f, 0.5f, 0.0f);

    YAML::Node scene = YAML::LoadFile("./resources/scenes/scene.yaml");

    auto objects = scene["objects"];

    for (auto i = 0; i < SPHERES - 1; i++) {
        auto object = objects[i];

        auto center = object["sphere"]["center"].as<Float3>();
        auto radius = object["sphere"]["radius"].as<Float>();
        auto materialId = object["sphere"]["materialId"].as<uint32_t>();
        createSphere(spheres, i, center, radius, *(materials[materialId]));
    }

    createPlane(spheres, 7, { 0.0f, 0.49f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.125f, 0.125f, 0.125f }, *(materials[8]));
#else
    auto eye = point(13.0f, 2.0f, 3.0f);
    auto center = point(0.0f, 0.0f, 0.0f);
    auto up = point(0.0f, 1.0f, 0.0f);
    auto focusDistance = 10.0f;
    auto aperture = 0.1f;
    camera->initialize(eye, center, up, Float(width) / height, 20.0f, aperture, focusDistance, 0.0f, 1.0f);

    // Scene2 Final
    for (auto a = -11; a < 11; a++) {
        for (auto b = -11; b < 11; b++) {
            auto index = (a + 11) * 22 + (b + 11);
            auto chooseMaterial = Utils::randomFloat();

            auto center = point(a + 0.9f * Utils::randomFloat(), 0.2f, b + 0.9f * Utils::randomFloat());

            if (length(center - point(4.0f, 0.2f, 0.0f)) > 0.9f) {
                if (chooseMaterial < 0.8f) {
                    // Diffuse
                    auto albedo = Color::random() * Color::random();
                    createLambertianMaterial(materials[index], albedo);
                    auto center1 = center + point(0.0f, Utils::randomFloat(0.0f, 0.5f), 0.0f);
                    createMovingSphere(spheres, index, center, center1, 0.0f, 1.0f, 0.2f, *(materials[index]));
                    //createSphere(spheres, index, center, 0.2f, *(materials[index]));

                }
                else if (chooseMaterial < 0.95f) {
                    // Metal
                    auto albedo = Color::random(0.5f, 1.0f);
                    auto fuzz = Utils::randomFloat(0.0f, 0.5f);
                    createMetalMaterial(materials[index], albedo, fuzz);
                    createSphere(spheres, index, center, 0.2f, *(materials[index]));
                }
                else {
                    // Glass
                    createDieletricMaterial(materials[index], 1.5f);
                    createSphere(spheres, index, center, 0.2f, *(materials[index]));
                }
            }
            else {
                auto albedo = Color::random() * Color::random();
                createLambertianMaterial(materials[index], albedo);
                createSphere(spheres, index, center, 0.2f, *(materials[index]));
            }
        }
    }

    createLambertianMaterial(materials[484], color(0.5f, 0.5f, 0.5f));
    createDieletricMaterial(materials[485], 1.5f);
    createLambertianMaterial(materials[486], color(0.4f, 0.2f, 0.1f), 1.0f);
    createMetalMaterial(materials[487], color(0.7f, 0.6f, 0.5f), 0.0f);

    createSphere(spheres, 484, point( 0.0f, -1000.0,  0.0f), 1000.0f, *(materials[484]));
    createSphere(spheres, 485, point( 0.0f,     1.0f, 0.0f),    1.0f, *(materials[485]));
    createSphere(spheres, 486, point(-4.0f,     1.0f, 0.0f),    1.0f, *(materials[486]));
    createSphere(spheres, 487, point( 4.0f,     1.0f, 0.0f),    1.0f, *(materials[487]));

#endif
    gpuErrorCheck(cudaMemcpyToSymbol(constantSpheres, spheres, sizeof(Sphere*) * SPHERES));

    auto pixelCount = width * height;
    randStates = createObjectArray<curandState>(pixelCount);

    renderInit<<<gridSize, blockSize>>>(width, height, randStates);
    gpuErrorCheck(cudaDeviceSynchronize());

    imageData = std::make_shared<ImageData>();

    imageData->width = width;
    imageData->height = height;
    imageData->channels = 3;
    imageData->size = pixelCount * 3;

    Utils::reportGPUUsageInfo();
}   

void clearBackBuffers() {
    clearBackBuffers<<<gridSize, blockSize>>>(canvas);
    gpuErrorCheck(cudaDeviceSynchronize());
    canvas->resetSampleCount();
    canvas->resetRenderingTime();
}

void pathTracing() {
#ifdef GPU_REALTIME
    if (camera->isDirty()) {
        clearBackBuffers();
        camera->updateViewMatrix();
        camera->resetDiryFlag();
    }

    canvas->incrementSampleCount();
    canvas->incrementRenderingTime(frameTime * 1000.0f);
    renderKernel<<<gridSize, blockSize>>>(canvas, camera, randStates, spheres, nullptr);
    gpuErrorCheck(cudaDeviceSynchronize());

    imageData->data = canvas->getPixelBuffer();
#else
    auto* counter = createObjectPtr<int32_t>();

    canvas->incrementSampleCount();
    renderKernel<<<gridSize, blockSize>>>(canvas, camera, randStates, spheres, counter);
    gpuErrorCheck(cudaDeviceSynchronize());

    deleteObject(counter);
#endif
}

void cleanup() {
    deleteObject(randStates);

    deleteDeviceObjectArray<<<1, 1>>>(spheres, SPHERES);

    for (auto i = 0; i < SPHERES; i++) {
        deleteDeviceObject<<<1, 1>>>(materials[i]);
        gpuErrorCheck(cudaDeviceSynchronize());
        gpuErrorCheck(cudaFree(materials[i]));
}

    deleteObject(spheres);

    deleteObject(camera);
    canvas->uninitialize();
    deleteObject(canvas);
}

#ifndef GPU_REALTIME
int main() {
    gpuErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    initialize(width, height);
    
    GPUTimer timer("Rendering start...");
    pathTracing();
    timer.stop("Rendering elapsed time");

    canvas->writeToPNG("render.png");
    Utils::openImage(L"render.png");

    cleanup();

    return 0;
}
#endif // !GPU_REALTIME