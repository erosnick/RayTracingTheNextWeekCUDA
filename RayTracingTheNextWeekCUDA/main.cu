
#include "main.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include "Utils.h"
#include "GPUTimer.h"
#include "Sphere.h"
#include "Plane.h"
#include "Triangle.h"
#include "Mesh.h"
#include "Cube.h"
#include "YAML.h"
#include "ModelLoader.h"
#include "kernels.h"

#include <yaml-cpp/yaml.h>
#include <cstdio>
#include <algorithm>

constexpr auto BOUNCES = 10;

constexpr auto OBJECTS = 9;
constexpr auto MATERIALS = 9;
CUDA_CONSTANT Hitable* constantObjects[OBJECTS];
CUDA_CONSTANT Material* constantMaterials[MATERIALS];
CUDA_CONSTANT Float3 constantFloats[5000];

CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) {
    HitResult tempHitResult;
    bool bHitAnything = false;
    Float closestSoFar = tMax;
    for (auto& object : constantObjects) {
        // Empty hit call costs ~130ms
        if (object->hit(ray, tMin, closestSoFar, tempHitResult)) {
            bHitAnything = true;
            closestSoFar = tempHitResult.t;
            hitResult = tempHitResult;
        }
    }

    return bHitAnything;
}

using ScatterFunction = bool (*)(const Ray& inRay, const HitResult& hitResult, Float3& attenuation, Ray& scattered, curandState* randState);

CUDA_DEVICE ScatterFunction scatterFunction;

CUDA_DEVICE Float3 rayColor(const Ray& ray, curandState* randState) {
    Ray currentRay = ray;
    auto currentAttenuation = make_float3(1.0f, 1.0f, 1.0f);
    for (auto i = 0; i < BOUNCES; i++) {
        HitResult hitResult;
        // Smaller tMin will has a impact on performance
        if (hit(currentRay, Math::epsilon, Math::infinity, hitResult)) {
            Float3 attenuation;
            Ray scattered;
            // Bounces 4 Samples 100 18ms
            // Bounces 4 Samples 100 33ms(Empty scatter function body)
            if (constantMaterials[hitResult.materialId]->scatter(currentRay, hitResult, attenuation, scattered, randState)) {
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
//            return currentAttenuation * attenuation;
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

CUDA_GLOBAL void renderKernel(Canvas* canvas, Camera* camera, curandState* randStates, int32_t* counter) {
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;
    auto width = canvas->getWidth();
    auto height = canvas->getHeight();
#ifdef GPU_REALTIME
    constexpr auto samplesPerPixel = 1;
#else
    constexpr auto samplesPerPixel = 32;
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
            color += rayColor(ray, &localRandState);
        }
        // Very important!!!
        randStates[index] = localRandState; 
#ifdef GPU_REALTIME
        canvas->accumulatePixel(index, color);
#else
        canvas->writePixel(index, color / samplesPerPixel);

        auto tenPercent = (width * height) / 10;

        auto old = atomicAdd(counter, 1);

        if ((old + 1) > 0 && (old + 1) % tenPercent == 0) {
            printf("Complete: %.2f%%\n", (float((old + 1) * 100) / (width * height)));
        }
#endif // GPU_REALTIME
    }
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
int32_t height = 1024;
#elif RESOLUTION == 3
int32_t width = 1024;
int32_t height = 576;
#elif RESOLUTION == 4
int32_t width = 1280;
int32_t height = 720;
#elif RESOLUTION == 5
int32_t width = 1920;
int32_t height = 1080;
#elif RESOLUTION == 6
int32_t width = 64;
int32_t height = 36;
#endif

#define SCENE 1

Canvas* canvas = nullptr;
Camera* camera = nullptr;
Hitable** spheres = nullptr;
Hitable** AABB = nullptr;
int32_t triangleCount = 0;
Material** materials = nullptr;
curandState* randStates = nullptr;
std::shared_ptr<ImageData> imageData = nullptr;
cudaTextureObject_t triangleDataTexture;
Float4* triangleData = nullptr;
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

cudaTextureObject_t createTriangleData(Float4* data, int32_t size) {
    cudaTextureObject_t texture;
    cudaResourceDesc textureResourceDesc;

    memset(&textureResourceDesc, 0, sizeof(cudaResourceDesc));

    auto desc = cudaCreateChannelDesc<Float4>();

    textureResourceDesc.resType = cudaResourceTypeLinear;
    textureResourceDesc.res.linear.devPtr = data;
    textureResourceDesc.res.linear.desc = desc;
    textureResourceDesc.res.linear.sizeInBytes = size;

    cudaTextureDesc textureDesc;
    memset(&textureDesc, 0, sizeof(cudaTextureDesc));

    textureDesc.normalizedCoords = false;
    textureDesc.filterMode = cudaFilterModeLinear;
    textureDesc.addressMode[0] = cudaAddressModeClamp;
    textureDesc.addressMode[1] = cudaAddressModeClamp;
    textureDesc.readMode = cudaReadModeElementType;

    gpuErrorCheck(cudaCreateTextureObject(&texture, &textureResourceDesc, &textureDesc, nullptr));

    testKernel<<<1, 1>>>(texture);

    gpuErrorCheck(cudaDeviceSynchronize());

    return texture;
}

void initialize(int32_t width, int32_t height) {
    gpuErrorCheck(cudaSetDevice(0));
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

    //for (auto& material : materials) {
    //    material = createObjectPtr<Material*>();
    //}

    materials = createObjectPtrArray<Material*>(MATERIALS);

    spheres = createObjectPtrArray<Hitable*>(OBJECTS);

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
    createDieletricMaterial(materials, 0, 1.5f);
    createDieletricMaterial(materials, 1, 1.5f);
    createLambertianMaterial(materials, 2, make_float3(0.1f, 0.2f, 0.5f));
    //createDieletricMaterial<<<1, 1>>>(materials[3], 1.5f);
    createMetalMaterial(materials, 3, make_float3(0.8f, 0.6f, 0.2f), 0.0f);
    //createLambertianMaterial<<<1, 1>>>(materials[4], make_float3(0.8f, 0.8f, 0.0f));
    createMetalMaterial(materials, 4, make_float3(0.5f, 0.7f, 1.0f), 0.0f);

    auto center1 = point(0.0f, 0.5f, 0.0f);

    //createSphere(spheres, 0, { -1.0f, 0.0f, -1.0f}, 0.5f, *(materials[0]));
    //createSphere(spheres, 1, { -1.0f, 0.0f, -1.0f }, -0.4f, *(materials[1]), false);
    //createSphere(spheres, 2, {  0.0f, 0.0f, -1.0f },  0.5f, *(materials[2]));
    //createSphere(spheres, 3, {  1.0f, 0.0f, -1.0f },  0.5f, *(materials[3]));
    //createSphere(spheres, 4, {  0.0f, -100.5f, -1.0f }, 100.0f, *(materials[4]));
    createSphere(spheres, 0, { -1.0f, 0.0f, -1.0f}, 0.5f, materials[0]);
    createSphere(spheres, 1, { -1.0f, 0.0f, -1.0f }, -0.4f, materials[1], false);
    createMovingSphere(spheres, 2, {  0.0f, 0.0f, -1.0f }, { 0.0f, 0.5f, -1.0f }, 0.0f, 1.0f, 0.5f, materials[2]);
    createSphere(spheres, 3, {  1.0f, 0.0f, -1.0f },  0.5f, materials[3]);
    createSphere(spheres, 4, {  0.0f, -100.5f, -1.0f }, 100.0f, materials[4]);
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
    //createDieletricMaterial(materials, 0, 1.5f);
    //createDieletricMaterial(materials, 1, 1.5f);
    //createLambertianMaterial(materials, 2, make_float3(0.1f, 0.2f, 0.5f));
    createLambertianMaterial(materials, 0, make_float3(1.0f, 0.0f, 0.0f));
    createLambertianMaterial(materials, 1, make_float3(0.0f, 1.0f, 0.0f));
    createLambertianMaterial(materials, 2, make_float3(0.0f, 0.0f, 1.0f));
    createLambertianMaterial(materials, 3, make_float3(1.0f, 1.0f, 1.0f));
    createLambertianMaterial(materials, 4, make_float3(0.75f, 0.25f, 0.25f));
    createLambertianMaterial(materials, 5, make_float3(0.25f, 0.25f, 0.75f));
    createMetalMaterial(materials, 6, make_float3(1.0f, 1.0f, 1.0f), 0.0f);
    createDieletricMaterial(materials, 7, 1.5f);
    createEmissionMaterial(materials, 8, make_float3(1.0f, 1.0f, 1.0f), 1.0f);
    //createDieletricMaterial(materials, 1, 1.5f);
    //createDieletricMaterial<<<1, 1>>>(materials[3], 1.5f);
    //createMetalMaterial(materials, 3, make_float3(0.8f, 0.6f, 0.2f), 0.0f);
    //createLambertianMaterial(materials, 4, make_float3(0.8f, 0.8f, 0.0f));
    //createMetalMaterial(materials, 4, make_float3(0.5f, 0.7f, 1.0f), 0.0f);

    //createSphere(spheres, 0, { -1.0f, 0.0f, -1.0f}, 0.5f, *(materials[0]));
    //createSphere(spheres, 1, { -1.0f, 0.0f, -1.0f }, -0.4f, *(materials[1]), false);
    //createSphere(spheres, 2, {  0.0f, 0.0f, -1.0f },  0.5f, *(materials[2]));
    //createSphere(spheres, 3, {  1.0f, 0.0f, -1.0f },  0.5f, *(materials[3]));
    //createSphere(spheres, 4, {  0.0f, -100.5f, -1.0f }, 100.0f, *(materials[4]));

    //auto model = loadModel("./resources/models/bunny/bunny.obj");
    //auto model = loadModel("./resources/models/cube/cube.obj");
    //auto model = loadModel("./resources/models/plane/plane.obj");
    //auto model = loadModel("./resources/models/test/test.obj");
    auto model = loadModel("./resources/models/suzanne/suzanne_small.obj");

    std::vector<Float> positionX;
    std::vector<Float> positionY;
    std::vector<Float> positionZ;

    for (const auto& vertex : model) {
        positionX.push_back(vertex.x);
        positionY.push_back(vertex.y);
        positionZ.push_back(vertex.z);
    }

    std::sort(positionX.begin(), positionX.end());
    std::sort(positionY.begin(), positionY.end());
    std::sort(positionZ.begin(), positionZ.end());

    Float3 minAABB = { positionX[0], positionY[0], positionZ[0] };
    Float3 maxAABB = { positionX[positionX.size()- 1],  positionY[positionY.size() - 1], positionZ[positionZ.size() - 1] };

    Float3 extendAABB = (maxAABB - minAABB) * 0.5f;
    Float3 centerAABB = (minAABB + maxAABB) * 0.5f;

    triangleCount = model.size() / 3;

    triangleData = createObjectArray<Float4>(model.size());

    for (auto i = 0; i < triangleCount; i++) {
        auto v0 = model[i * 3];
        auto v1 = model[i * 3 + 1];
        auto v2 = model[i * 3 + 2];
        auto E1 = v1 - v0;
        auto E2 = v2 - v0;
        auto normal = normalize(cross(E1, E2));
        triangleData[i * 3] = make_float4(v0.x, v0.y, v0.z, 0.0f);
        triangleData[i * 3 + 1] = make_float4(v1.x, v1.y, v1.z, 0.0f);
        triangleData[i * 3 + 2] = make_float4(v2.x, v2.y, v2.z, 0.0f);
    }

    AABB = createObjectPtr<Hitable*>();

    createCube(AABB, 0, centerAABB, extendAABB, materials[0]);

    triangleDataTexture = createTriangleData(triangleData, sizeof(Float4) * model.size());

    //createCube(spheres, 1, centerAABB, extendAABB, materials[0]);
    createPlane(spheres, 0, {  0.0f,  0.5f,  0.0f }, {  0.0f,  1.0f,  0.0f }, { 0.5f, 0.5f, 0.5f }, materials[3], PlaneOrientation::XZ);
    createPlane(spheres, 1, {  0.0f, -0.5f,  0.0f }, {  0.0f, -1.0f,  0.0f }, { 0.5f, 0.5f, 0.5f }, materials[3], PlaneOrientation::XZ);
    createPlane(spheres, 2, { -0.5f,  0.0f,  0.0f }, { -1.0f,  0.0f,  0.0f }, { 0.5f, 0.5f, 0.5f }, materials[4], PlaneOrientation::YZ);
    createPlane(spheres, 3, {  0.5f,  0.0f,  0.0f }, {  1.0f,  0.0f,  0.0f }, { 0.5f, 0.5f, 0.5f }, materials[5], PlaneOrientation::YZ);
    createPlane(spheres, 4, {  0.0f,  0.0f, -0.5f }, {  0.0f,  0.0f, -1.0f }, { 0.5f, 0.5f, 0.5f }, materials[3], PlaneOrientation::XY);
    createSphere(spheres, 5, { -0.225f, -0.325f, -0.25f }, 0.175f, materials[6]);
    createSphere(spheres, 6, { 0.275f, -0.325f, -0.125f }, 0.175f, materials[7]);
    createPlane(spheres, 7, { 0.0f,  0.49f,  0.0f }, { 0.0f,  1.0f,  0.0f }, { 0.125f, 0.125f, 0.125f }, materials[8], PlaneOrientation::XZ, false);
    createMesh(spheres, 8, triangleCount, triangleDataTexture, minAABB, maxAABB, materials[3]);
    //createSphere(spheres, 5, { 0.0f, -102.0f, -1.0f }, 100.0f, materials[4]);

#elif SCENE == 2
    // If the distance between object and camera equals to focus lens
    // then the object is in focus
    YAML::Node scene = YAML::LoadFile("./resources/scenes/cornellbox.yaml");

    auto eye = scene["camera"]["eye"].as<Float3>();
    auto center = scene["camera"]["center"].as<Float3>();
    auto up = scene["camera"]["up"].as<Float3>();
    auto focusDistance = length(center - eye);
    auto aperture = scene["camera"]["aperture"].as<Float>();
    auto fov = scene["camera"]["fov"].as<Float>();
    camera->initialize(eye, center, up, Float(width) / height, fov, aperture, focusDistance, 0.0f, 1.0f);

    //// Scene1 Defocus Blur
    ////createLambertianMaterial(materials[0], make_float3(0.0f, 1.0f, 0.0f));
    ////createLambertianMaterial(materials[1], make_float3(1.0f, 0.0f, 0.0f));
    //createLambertianMaterial(materials, 2, make_float3(1.0f, 1.0f, 1.0f));
    //createDieletricMaterial(materials, 3, 1.5f);
    //createMetalMaterial(materials, 4, make_float3(0.8f, 0.6f, 0.2f), 0.0f);
    //createMetalMaterial(materials, 4, make_float3(1.0f, 1.0f, 1.0f), 0.0f);
    //createLambertianMaterial(materials, 5, make_float3(0.8f, 0.8f, 0.0f));
    //createMetalMaterial(materials, 6, make_float3(0.5f, 0.7f, 1.0f), 0.0f);

    ////createLambertianMaterial(materials[5], make_float3(1.0f, 1.0f, 1.0f));
    //createDieletricMaterial(materials, 7, 1.5f);
    //createDieletricMaterial(materials, 8, 1.5f);
    //createEmissionMaterial(materials, 5, make_float3(1.0f, 1.0f, 1.0f), 10.0f);

    auto objects = scene["objects"];

    for (auto i = 0; i < OBJECTS; i++) {
        // 场景的构成是objects是几何体数组
        // 数组的元素是Map，其中又包含若干几何体属性Map
        // objects:
        //    -sphere : # Left
        //        type : 0
        //        center :
        //          - -1000.5
        //          -  0.0
        //          -  0.0
        //        radius : 1000
        //        materialId : 1
        //        material :
        //        type : 0
        //        albedo :
        //          - 0.75
        //          - 0.25
        //          - 0.25
        // 表示Map中第一个元素的迭代器，这里sphere是一个Map
        // 这里的key就是字符串"sphere"
        auto iterator = objects[i].begin();
        auto key = iterator->first.as<std::string>();

        auto object = objects[i][key];

        auto materialType = static_cast<MaterialType>(object["material"]["type"].as<uint8_t>());

        iterator = object.begin();

        auto materialId = object["materialId"].as<uint32_t>();

        switch (materialType) {
            case MaterialType::Lambertian: {
                auto albedo = object["material"]["albedo"].as<Float3>();

                if ((materials[materialId]) == nullptr) {
                    createLambertianMaterial(materials, materialId, albedo);
                }
            }
        
            break;
            case MaterialType::Dieletric: {
                auto indexOfRefraction = object["material"]["indexOfRefraction"].as<Float>();

                if ((materials[materialId]) == nullptr) {
                    createDieletricMaterial(materials, materialId, indexOfRefraction);
                }
            }
                                        
            break;
            case MaterialType::Metal: {
                auto albedo = object["material"]["albedo"].as<Float3>();
                auto fuzz = object["material"]["fuzz"].as<Float>();

                if ((materials[materialId]) == nullptr) {
                    createMetalMaterial(materials, materialId, albedo, fuzz);
                }
            }

            break;
            case MaterialType::Emission: {
                auto albedo = object["material"]["albedo"].as<Float3>();
                auto intensity = object["material"]["intensity"].as<Float>();

                if ((materials[materialId]) == nullptr) {
                    createEmissionMaterial(materials, materialId, albedo, intensity);
                }
            }
        }

        auto primitiveType = static_cast<PrimitiveType>(iterator->second.as<uint8_t>());

        switch (primitiveType) {
            case PrimitiveType::Sphere: {
                auto center = object["center"].as<Float3>();
                auto radius = object["radius"].as<Float>();

                createSphere(spheres, i, center, radius, materials[materialId]);
            }
            
            break;
            case PrimitiveType::Plane: {
                auto position = object["position"].as<Float3>();
                auto normal = object["normal"].as<Float3>();
                auto extend = object["extend"].as<Float3>();

                createPlane(spheres, i, position, normal, extend, materials[materialId], PlaneOrientation::XZ);
            }

            break;
            case PrimitiveType::Triangle: {
                auto v0 = object["v0"].as<Float3>();
                auto v1 = object["v1"].as<Float3>();
                auto v2 = object["v2"].as<Float3>();

                createTriangle(spheres, i, v0, v1, v2, materials[materialId]);
            }

            break;
        }
    }

    //// If the distance between object and camera equals to focus lens
    //// then the object is in focus
    //auto eye = point(0.0f, 0.0f, 1.5f);
    //auto center = point(0.0f, 0.0f, -1.0f);
    //auto up = point(0.0f, 1.0f, 0.0f);
    //auto focusDistance = length(center - eye);
    //auto aperture = 0.0f;
    //camera->initialize(eye, center, up, Float(width) / height, 45.0f, aperture, focusDistance, 0.0f, 1.0f);

    //// Scene1 Defocus Blur
    //createDieletricMaterial(materials[0], 1.5f);
    //createDieletricMaterial(materials[1], 1.5f);
    //createLambertianMaterial(materials[2], make_float3(0.1f, 0.2f, 0.5f));
    ////createDieletricMaterial<<<1, 1>>>(materials[3], 1.5f);
    //createMetalMaterial(materials[3], make_float3(0.8f, 0.6f, 0.2f), 0.0f);
    ////createLambertianMaterial<<<1, 1>>>(materials[4], make_float3(0.8f, 0.8f, 0.0f));
    //createMetalMaterial(materials[4], make_float3(0.5f, 0.7f, 1.0f), 0.0f);
    //createLambertianMaterial(materials, 5, make_float3(1.0f, 1.0f, 1.0f));

    ////createSphere(spheres, 0, { -1.0f, 0.0f, -1.0f}, 0.5f, *(materials[0]));
    ////createSphere(spheres, 1, { -1.0f, 0.0f, -1.0f }, -0.4f, *(materials[1]), false);
    ////createSphere(spheres, 2, {  0.0f, 0.0f, -1.0f },  0.5f, *(materials[2]));
    ////createSphere(spheres, 3, {  1.0f, 0.0f, -1.0f },  0.5f, *(materials[3]));
    ////createSphere(spheres, 4, {  0.0f, -100.5f, -1.0f }, 100.0f, *(materials[4]));
    ////createSphere(spheres, 0, { -1.0f, 0.0f, -1.0f }, 0.5f, *(materials[5]));
    ////createSphere(spheres, 1, { -1.0f, 0.0f, -1.0f }, -0.4f, *(materials[1]), false);
    ////createSphere(spheres, 2, { 0.0f, 0.0f, -1.0f }, 0.5f, *(materials[5]));
    //createSphere(spheres, 0, {  0.25f, -0.325f, -0.125f }, 0.175f, *(materials[5]));
    //createSphere(spheres, 1, { -0.25f, -0.325f, -0.25f }, 0.175f, * (materials[5]));
    //createSphere(spheres, 2, {  0.0f, -1000.5f, 0.0f }, 1000.0f, *(materials[5]));
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
    gpuErrorCheck(cudaMemcpyToSymbol(constantObjects, spheres, sizeof(Hitable*) * OBJECTS));
    gpuErrorCheck(cudaMemcpyToSymbol(constantMaterials, materials, sizeof(Material*) * MATERIALS));

    auto pixelCount = width * height;
    randStates = createObjectArray<curandState>(pixelCount);

    renderInit<<<gridSize, blockSize>>>(width, height, randStates);
    gpuErrorCheck(cudaDeviceSynchronize());

    imageData = std::make_shared<ImageData>();

    imageData->width = width;
    imageData->height = height;
    imageData->channels = 3;
    imageData->size = pixelCount * 3;
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
    renderKernel<<<gridSize, blockSize>>>(canvas, camera, randStates, nullptr);
    gpuErrorCheck(cudaDeviceSynchronize());

    imageData->data = canvas->getPixelBuffer();
#else
    auto* counter = createObjectPtr<int32_t>();
    (*counter) = 0;
    canvas->incrementSampleCount();
    renderKernel<<<gridSize, blockSize>>>(canvas, camera, randStates, counter);
    gpuErrorCheck(cudaDeviceSynchronize());

    deleteObject(counter);
#endif
}

void cleanup() {
    deleteObject(randStates);

    deleteDeviceObjectArray<<<1, 1>>>(spheres, OBJECTS);
    deleteDeviceObjectArray<<<1, 1>>>(materials, MATERIALS);

    gpuErrorCheck(cudaDeviceSynchronize());

    gpuErrorCheck(cudaFree(triangleData));
    gpuErrorCheck(cudaDestroyTextureObject(triangleDataTexture));
    deleteObject(AABB);
    deleteObject(spheres);
    deleteObject(materials);

    deleteObject(camera);
    canvas->uninitialize();
    deleteObject(canvas);

    cudaDeviceReset();

    Utils::reportGPUUsageInfo();
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