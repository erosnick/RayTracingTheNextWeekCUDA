
#include "main.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include "Utils.h"
#include "GPUTimer.h"
#include "Sphere.h"
#include "Plane.h"
#include "TriangleMesh.h"
#include "Cube.h"
#include "YAML.h"
#include "ModelLoader.h"
#include "kernels.h"

#include "CUDAPathTracer.h"
#include "Loader.h"

#include <yaml-cpp/yaml.h>
#include <cstdio>
#include <algorithm>

constexpr auto BOUNCES = 10;

constexpr auto OBJECTS = 10;
constexpr auto MATERIALS = 9;
CUDA_CONSTANT Hitable* constantObjects[OBJECTS];
CUDA_CONSTANT Material* constantMaterials[MATERIALS];

// CUDA arrays
struct Payload {
    Vertex* cudaVertices2 = nullptr;
    Triangle* cudaTriangles2 = nullptr;
    float* cudaTriangleIntersectionData2 = nullptr;
    int* cudaTriIdxList2 = nullptr;
    float* cudaBVHlimits2 = nullptr;
    int* cudaBVHindexesOrTrilists2 = nullptr;
    TriangleDataTexture triangleDataTexture;
};

Payload payload;

CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) {
    HitResult tempHitResult;
    bool bHitAnything = false;
    Float closestSoFar = tMax;
    for (auto& object : constantObjects) {
        // Empty hit call costs ~130ms
        if (object && object->hit(ray, tMin, closestSoFar, tempHitResult)) {
            bHitAnything = true;
            closestSoFar = tempHitResult.t;
            hitResult = tempHitResult;
        }
    }

    return bHitAnything;
}

using ScatterFunction = bool (*)(const Ray& inRay, const HitResult& hitResult, Vector3Df& attenuation, Ray& scattered, curandState* randState);

CUDA_DEVICE ScatterFunction scatterFunction;

CUDA_DEVICE Vector3Df rayColor(const Ray& ray, curandState* randState) {
    Ray currentRay = ray;
    auto currentAttenuation = Vector3Df(1.0f, 1.0f, 1.0f);
    for (auto i = 0; i < BOUNCES; i++) {
        HitResult hitResult;
        // Smaller tMin will has a impact on performance
        if (hit(currentRay, Math::epsilon, Math::infinity, hitResult)) {
            Vector3Df attenuation;
            Ray scattered;
            // Bounces 4 Samples 100 18ms
            // Bounces 4 Samples 100 33ms(Empty scatter function body)
            //if (constantMaterials[hitResult.materialId]->scatter(currentRay, hitResult, attenuation, scattered, randState)) {
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
            auto background = lerp(Vector3Df(1.0f, 1.0f, 1.0f), Vector3Df(0.5f, 0.7f, 1.0f), t);
            return currentAttenuation * background;
        }
    }
    // exceeded recursion
    return Vector3Df(0.0f, 0.0f, 0.0f);
}

//CUDA_DEVICE Vector3Df rayColor(const Ray& ray, curandState* randState, Sphere* spheres, int32_t depth) {
//    if (depth == 0) {
//        // exceeded recursion
//        return make_float3(0.0f, 0.0f, 0.0f);
//    }
//    HitResult hitResult;
//    // Smaller tMin will has a impact on performance
//    if (hit(ray, Math::epsilon, Math::infinity, hitResult, spheres)) {
//        Vector3Df attenuation;
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

CUDA_GLOBAL void renderKernel(Canvas* canvas, Camera* camera, curandState* randStates, int32_t* counter, Payload payload) {
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
        auto color = Vector3Df(0.0f, 0.0f, 0.0f);
        auto localRandState = randStates[index];
        for (auto i = 0; i < samplesPerPixel; i++) {

            auto rx = curand_uniform(&localRandState);
            auto ry = curand_uniform(&localRandState);

            auto dx = Float(x + rx) / (width - 1);
            auto dy = Float(y + ry) / (height - 1);

            auto ray = camera->getRay(dx, dy, &localRandState);
            //color += rayColor(ray, &localRandState);
            color += pathTrace(&localRandState, ray.origin, ray.direction, -1, payload.cudaTriangles2, payload.cudaBVHindexesOrTrilists2, 
                                payload.cudaBVHlimits2, payload.cudaTriangleIntersectionData2, payload.cudaTriIdxList2, payload.triangleDataTexture);
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
constexpr auto MESHES = 2;
std::vector<TriangleMeshData> triangleMeshData;
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

void computeMeshAABB(const std::vector<Vector3Df>& vertices, Vector3Df& minAABB, Vector3Df& maxAABB) {
    std::vector<Float> positionX;
    std::vector<Float> positionY;
    std::vector<Float> positionZ;

    for (const auto& vertex : vertices) {
        positionX.push_back(vertex.x);
        positionY.push_back(vertex.y);
        positionZ.push_back(vertex.z);
    }

    std::sort(positionX.begin(), positionX.end());
    std::sort(positionY.begin(), positionY.end());
    std::sort(positionZ.begin(), positionZ.end());

    minAABB = { positionX[0], positionY[0], positionZ[0] };
    maxAABB = { positionX[positionX.size() - 1],  positionY[positionY.size() - 1], positionZ[positionZ.size() - 1] };

    Vector3Df extendAABB = (maxAABB - minAABB) * 0.5f;
    Vector3Df centerAABB = (minAABB + maxAABB) * 0.5f;
}

//std::vector<AABBox> computeTriangleAABBs(const std::vector<Vector3Df>& vertices) {
//    std::vector<AABBox> AABBs;
//
//    for (auto i = 0; i < vertices.size() / 3; i++) {
//        auto v0 = vertices[i * 3];
//        auto v1 = vertices[i * 3 + 1];
//        auto v2 = vertices[i * 3 + 2];
//
//        auto min = minf3(minf3(v0, v1), v2);
//        auto max = maxf3(maxf3(v0, v1), v2);
//
//        AABBs.emplace_back(AABBox(min, max));
//    }
//
//    return AABBs;
//}

std::vector<Vector3Df> computeTriangleAABBs(const std::vector<Vector3Df>& vertices) {
    std::vector<Vector3Df> AABBs;

    for (auto i = 0; i < vertices.size() / 3; i++) {
        auto v0 = vertices[i * 3];
        auto v1 = vertices[i * 3 + 1];
        auto v2 = vertices[i * 3 + 2];

        auto min = min3(min3(v0, v1), v2);
        auto max = max3(max3(v0, v1), v2);

        AABBs.emplace_back(Vector3Df(min.x, min.y, min.z));
        AABBs.emplace_back(Vector3Df(max.x, max.y, max.z));
    }

    return AABBs;
}

void prepareTriangleData(const std::vector<Vector3Df>& vertices, Float4* triangleData, Float4* aabbData) {
    triangleCount = vertices.size() / 3;

    //for (auto i = 0; i < triangleCount; i++) {
    //    auto v0 = vertices[i * 3];
    //    auto v1 = vertices[i * 3 + 1];
    //    auto v2 = vertices[i * 3 + 2];
    //    auto E1 = v1 - v0;
    //    auto E2 = v2 - v0;
    //    auto normal = normalize(cross(E1, E2));
    //    triangleData[i * 3] = make_float4(v0.x, v0.y, v0.z, normal.x);
    //    triangleData[i * 3 + 1] = make_float4(v1.x, v1.y, v1.z, normal.y);
    //    triangleData[i * 3 + 2] = make_float4(v2.x, v2.y, v2.z, normal.z);
    //}

    // Precompute E1, E2, store per triangle data as v0, E1, E2
    for (auto i = 0; i < triangleCount; i++) {
        auto v0 = vertices[i * 3];
        auto v1 = vertices[i * 3 + 1];
        auto v2 = vertices[i * 3 + 2];
        auto E1 = v1 - v0;
        auto E2 = v2 - v0;
        auto normal = normalize(cross(E1, E2));
        triangleData[i * 3] = make_float4(v0.x, v0.y, v0.z, normal.x);
        triangleData[i * 3 + 1] = make_float4(E1.x, E1.y, E1.z, normal.y);
        triangleData[i * 3 + 2] = make_float4(E2.x, E2.y, E2.z, normal.z);

        auto min = min3(min3(v0, v1), v2);
        auto max = max3(max3(v0, v1), v2);
        aabbData[i * 2] = make_float4(min.x, min.y, min.z, 0.0f);
        aabbData[i * 2 + 1] = make_float4(max.x, max.y, max.z, 0.0f);
    }
}

template<typename T1, typename T2 = T1>
cudaTextureObject_t createTextureObject(T1* data, int32_t size) {
    cudaTextureObject_t texture;
    cudaResourceDesc textureResourceDesc;

    memset(&textureResourceDesc, 0, sizeof(cudaResourceDesc));
    
    // Helper function to create cudaChannelFormatDesc
    auto desc = cudaCreateChannelDesc<T2>();

    // 1D texture array
    textureResourceDesc.resType = cudaResourceTypeLinear;
    textureResourceDesc.res.linear.devPtr = data;           // Create by cudaMallocManaged
    textureResourceDesc.res.linear.desc = desc;
    textureResourceDesc.res.linear.sizeInBytes = size;      // Data size, should be aligned to cudaDeviceProp::textureAlignment

    cudaTextureDesc textureDesc;
    memset(&textureDesc, 0, sizeof(cudaTextureDesc));

    // cudaAddressModeWrap and cudaAddressModeMirror are
    // only supported for normalized texture coordinates
    textureDesc.normalizedCoords = false;
    textureDesc.filterMode = cudaFilterModePoint;
    textureDesc.addressMode[0] = cudaAddressModeClamp;
    textureDesc.addressMode[1] = cudaAddressModeClamp;
    textureDesc.readMode = cudaReadModeElementType;

    gpuErrorCheck(cudaCreateTextureObject(&texture, &textureResourceDesc, &textureDesc, nullptr));

    return texture;
}

void createTriangleMeshData(const std::vector<Vector3Df>& vertices, TriangleMeshData& triangleMeshData) {
    triangleMeshData.triangleCount = vertices.size() / 3;

    computeMeshAABB(vertices, triangleMeshData.boundsMin, triangleMeshData.boundsMax);
    triangleMeshData.vertices = createObjectArray<Float4>(vertices.size());
    triangleMeshData.AABBs = createObjectArray<Float4>(vertices.size() * 2);

    prepareTriangleData(vertices, triangleMeshData.vertices, triangleMeshData.AABBs);

    // cudaDeviceProp::textureAlignment = 512
    auto dataSize = sizeof(Float4) * vertices.size();
    auto alignedElementCount = (dataSize + 512 - 1) / 512;
    triangleMeshData.triangleData = createTextureObject(triangleMeshData.vertices, alignedElementCount * 512);

    dataSize = sizeof(Float4) * vertices.size() * 2;
    alignedElementCount = (dataSize + 512 - 1) / 512;
    triangleMeshData.AABBData = createTextureObject(triangleMeshData.AABBs, alignedElementCount * 512);
}

// initialises scene data, builds BVH
void prepareCUDAscene() {

    // specify scene filename 
    //const char* scenefile = "data/teapot.ply";  // teapot.ply, big_atc.ply
    //const char* scenefile = "data/bunny.obj";
    //const char* scenefile = "data/bun_zipper_res2.ply";  // teapot.ply, big_atc.ply
    //const char* scenefile = "data/bun_zipper.ply";  // teapot.ply, big_atc.ply
    //const char* sceneFile = "./resources/models/dragon/dragon_vrip_res4.ply";  // teapot.ply, big_atc.ply
    //const char* scenefile = "data/dragon_vrip.ply";  // teapot.ply, big_atc.ply
    //const char* scenefile = "data/happy_vrip.ply";  // teapot.ply, big_atc.ply
    //auto sceneFile = "./resources/models/bunny/bunny.ply";
    //auto sceneFile = "./resources/models/suzanne/suzanne0.ply";

    //// load scene
    //loadObject(sceneFile, ReflectionType::COAT);

    //sceneFile = "./resources/models/suzanne/suzanne1.ply";

    //loadObject(sceneFile, ReflectionType::REFRACTION);

    //sceneFile = "./resources/models/suzanne/suzanne2.ply";

    //loadObject(sceneFile, ReflectionType::METAL);

    auto sceneFile = "./resources/models/materialball/materialball.ply";

    loadObject(sceneFile, ReflectionType::METAL, 0);

    float maxi = processTriangleData(Vector3Df(0.1f, 0.0f, -1.0f));

    // build the BVH
    UpdateBoundingVolumeHierarchy(sceneFile);

    // now, allocate the CUDA side of the data (in CUDA global memory,
    // in preparation for the textures that will store them...)

    // store vertices in a GPU friendly format using float4
    // copy vertex data to CUDA global memorya
    cudaMallocManaged(&payload.cudaVertices2, (g_verticesNo * 8 * sizeof(float)));

    for (unsigned f = 0; f < g_verticesNo; f++) {

        // first float4 stores vertex xyz position and precomputed ambient occlusion
        payload.cudaVertices2[f].x = g_vertices[f].x;
        payload.cudaVertices2[f].y = g_vertices[f].y;
        payload.cudaVertices2[f].z = g_vertices[f].z;
        payload.cudaVertices2[f]._ambientOcclusionCoeff = g_vertices[f]._ambientOcclusionCoeff;
        // second float4 stores vertex normal xyz
        payload.cudaVertices2[f]._normal.x = g_vertices[f]._normal.x;
        payload.cudaVertices2[f]._normal.y = g_vertices[f]._normal.y;
        payload.cudaVertices2[f]._normal.z = g_vertices[f]._normal.z;
    }

    // store precomputed triangle intersection data in a GPU friendly format using float4
    // copy precomputed triangle intersection data to CUDA global memory
    auto dataElementCount = 24;
    cudaMallocManaged(&payload.cudaTriangleIntersectionData2, g_trianglesNo * dataElementCount * sizeof(float));

    for (unsigned e = 0; e < g_trianglesNo; e++) {
        // Texture-wise:
        //
        // first float4, triangle center + two-sided bool
        auto index = 0;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._center.x;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._center.y;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._center.z;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._twoSided ? 1.0f : 0.0f;
        // second float4, normal
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._normal.x;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._normal.y;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._normal.z;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._d;
        // third float4, precomputed plane normal of triangle edge 1
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._e1.x;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._e1.y;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._e1.z;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._d1;
        // fourth float4, precomputed plane normal of triangle edge 2
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._e2.x;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._e2.y;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._e2.z;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._d2;
        // fifth float4, precomputed plane normal of triangle edge 3
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._e3.x;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._e3.y;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._e3.z;
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e]._d3;

        // extra info
        payload.cudaTriangleIntersectionData2[dataElementCount * e + index++] = g_triangles[e].materialType;
    }

    // copy triangle data to CUDA global memory
    cudaMalloc((void**)&payload.cudaTriangles2, g_trianglesNo * sizeof(Triangle));
    cudaMemcpy(payload.cudaTriangles2, &g_triangles[0], g_trianglesNo * sizeof(Triangle), cudaMemcpyHostToDevice);

    // Allocate CUDA-side data (global memory for corresponding textures) for Bounding Volume Hierarchy data
    // See BVH.h for the data we are storing (from CacheFriendlyBVHNode)

    // Leaf nodes triangle lists (indices to global triangle list)
    // copy triangle indices to CUDA global memory
    cudaMalloc((void**)&payload.cudaTriIdxList2, g_triIndexListNo * sizeof(int));
    cudaMemcpy(payload.cudaTriIdxList2, g_triIndexList, g_triIndexListNo * sizeof(int), cudaMemcpyHostToDevice);

    // Bounding box limits need bottom._x, top._x, bottom._y, top._y, bottom._z, top._z...
    // store BVH bounding box limits in a GPU friendly format using float2

    // copy BVH limits to CUDA global memory
    cudaMallocManaged(&payload.cudaBVHlimits2, g_pCFBVH_No * 6 * sizeof(float));

    for (unsigned h = 0; h < g_pCFBVH_No; h++) {
        // Texture-wise:
        // First float2
        payload.cudaBVHlimits2[6 * h + 0] = g_pCFBVH[h]._bottom.x;
        payload.cudaBVHlimits2[6 * h + 1] = g_pCFBVH[h]._top.x;
        // Second float2
        payload.cudaBVHlimits2[6 * h + 2] = g_pCFBVH[h]._bottom.y;
        payload.cudaBVHlimits2[6 * h + 3] = g_pCFBVH[h]._top.y;
        // Third float2
        payload.cudaBVHlimits2[6 * h + 4] = g_pCFBVH[h]._bottom.z;
        payload.cudaBVHlimits2[6 * h + 5] = g_pCFBVH[h]._top.z;
    }

    // ..and finally, from CacheFriendlyBVHNode, the 4 integer values:
    // store BVH node attributes (triangle count, startindex, left and right child indices) in a GPU friendly format using uint4

    // copy BVH node attributes to CUDA global memory
    cudaMallocManaged(&payload.cudaBVHindexesOrTrilists2, g_pCFBVH_No * 4 * sizeof(unsigned));

    for (unsigned g = 0; g < g_pCFBVH_No; g++) {
        // Texture-wise:
        // A single uint4
        payload.cudaBVHindexesOrTrilists2[4 * g + 0] = g_pCFBVH[g].u.leaf._count;  // number of triangles stored in this node if leaf node
        payload.cudaBVHindexesOrTrilists2[4 * g + 1] = g_pCFBVH[g].u.inner._idxRight; // index to right child if inner node
        payload.cudaBVHindexesOrTrilists2[4 * g + 2] = g_pCFBVH[g].u.inner._idxLeft;  // index to left node if inner node
        payload.cudaBVHindexesOrTrilists2[4 * g + 3] = g_pCFBVH[g].u.leaf._startIndexInTriIndexList; // start index in list of triangle indices if leaf node
        // union
    }

    payload.triangleDataTexture.triIdxListTexture = createTextureObject(payload.cudaTriIdxList2, g_triIndexListNo * sizeof(uint1));

    payload.triangleDataTexture.CFBVHlimitsTexture = createTextureObject<float, Float2>(payload.cudaBVHlimits2, g_pCFBVH_No * 6 * sizeof(float));

    payload.triangleDataTexture.CFBVHindexesOrTrilistsTexture = createTextureObject<int, uint4>(payload.cudaBVHindexesOrTrilists2, g_pCFBVH_No * sizeof(uint4));

    payload.triangleDataTexture.trianglesTexture = createTextureObject<float, Float4>(payload.cudaTriangleIntersectionData2, g_trianglesNo * dataElementCount * sizeof(float));

    // Initialisation Done!
    std::cout << "Rendering data initialised and copied to CUDA global memory\n";
}

void initialize(int32_t width, int32_t height) {
    gpuErrorCheck(cudaSetDevice(0));
    //Canvas canvas(width, height);
    Utils::reportGPUUsageInfo();
    Utils::queryDeviceProperties();
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
    //YAML::Node scene = YAML::LoadFile("./resources/scenes/cornellbox_empty.yaml");
    //YAML::Node scene = YAML::LoadFile("./resources/scenes/cornellbox0.yaml");
    //YAML::Node scene = YAML::LoadFile("./resources/scenes/cornellbox1.yaml");
    YAML::Node scene = YAML::LoadFile("./resources/scenes/cornellbox2.yaml");
    //YAML::Node scene = YAML::LoadFile("./resources/scenes/cornellbox3.yaml");

    auto eye = scene["camera"]["eye"].as<Vector3Df>();
    auto center = scene["camera"]["center"].as<Vector3Df>();
    auto up = scene["camera"]["up"].as<Vector3Df>();
    auto focusDistance = (center - eye).length();
    auto aperture = scene["camera"]["aperture"].as<Float>();
    auto fov = scene["camera"]["fov"].as<Float>();
    camera->initialize(eye, center, up, Float(width) / height, fov, aperture, focusDistance, 0.0f, 1.0f);
    // Scene1 Defocus Blur
    //createDieletricMaterial(materials, 0, 1.5f);
    //createDieletricMaterial(materials, 1, 1.5f);
    //createLambertianMaterial(materials, 2, make_float3(0.1f, 0.2f, 0.5f));
    createLambertianMaterial(materials, 0, Vector3Df(1.0f, 0.0f, 0.0f));
    createLambertianMaterial(materials, 1, Vector3Df(0.0f, 1.0f, 0.0f));
    createLambertianMaterial(materials, 2, Vector3Df(0.0f, 0.0f, 1.0f));
    createLambertianMaterial(materials, 3, Vector3Df(1.0f, 1.0f, 1.0f));
    createLambertianMaterial(materials, 4, Vector3Df(0.75f, 0.25f, 0.25f));
    createLambertianMaterial(materials, 5, Vector3Df(0.25f, 0.25f, 0.75f));
    createMetalMaterial(materials, 6, Vector3Df(1.0f, 1.0f, 1.0f), 0.0f);
    createDieletricMaterial(materials, 7, 1.5f);
    createEmissionMaterial(materials, 8, Vector3Df(1.0f, 1.0f, 1.0f), 5.0f);
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

    //auto meshData = loadModel("./resources/models/bunny/bunny.obj");
    auto meshData = loadModel("./resources/models/cube/cube_small.obj", Vector3Df(0.5f, 1.0f, 0.5f), Vector3Df(0.0f, 30.0f, 0.0f), Vector3Df(-0.25f, -0.25f, -0.25f));
    auto vertices = meshData.vertices;

    //meshData = loadModel("./resources/models/cube/cube_small.obj", make_float3(0.5f, 0.5f, 0.5f), make_float3(0.0f, -30.0f, 0.0f), make_float3(0.25f, -0.375f, -0.25f));
    //meshDatas.push_back(meshData);
    //auto meshData = loadModel("./resources/models/sphere/sphere.obj");
    //auto meshData = loadModel("./resources/models/cube/cuboid.obj");
    //auto model = loadModel("./resources/models/plane/plane.obj");
    //auto model = loadModel("./resources/models/test/test.obj");
    //auto meshData = loadModel("./resources/models/suzanne/suzanne_small.obj");
    prepareCUDAscene();

    auto objects = scene["objects"];

    for (auto i = 0; i < objects.size(); i++) {
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
                auto albedo = object["material"]["albedo"].as<Vector3Df>();

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
                auto albedo = object["material"]["albedo"].as<Vector3Df>();
                auto fuzz = object["material"]["fuzz"].as<Float>();

                if ((materials[materialId]) == nullptr) {
                    createMetalMaterial(materials, materialId, albedo, fuzz);
                }
            }

            break;
            case MaterialType::Emission: {
                auto albedo = object["material"]["albedo"].as<Vector3Df>();
                auto intensity = object["material"]["intensity"].as<Float>();

                if ((materials[materialId]) == nullptr) {
                    createEmissionMaterial(materials, materialId, albedo, intensity);
                }
            }
        }

        auto primitiveType = static_cast<PrimitiveType>(iterator->second.as<uint8_t>());

        switch (primitiveType) {
            case PrimitiveType::Sphere: {
                auto center = object["center"].as<Vector3Df>();
                auto radius = object["radius"].as<Float>();

                createSphere(spheres, i, center, radius, materials[materialId]);
            }

            break;
            case PrimitiveType::Plane: {
                auto position = object["position"].as<Vector3Df>();
                auto normal = object["normal"].as<Vector3Df>();
                auto extend = object["extend"].as<Vector3Df>();
                auto twoSide = object["twoSide"].as<bool>();
                auto orientation = static_cast<PlaneOrientation>(object["orientation"].as<uint8_t>());
                createPlane(spheres, i, position, normal, extend, materials[materialId], orientation, twoSide);
            }

            break;
            case PrimitiveType::TriangleMesh: {
                auto path = "./resources/models/" + object["model"].as<std::string>();
                auto scale = object["scale"].as<Vector3Df>();
                auto rotate = object["rotate"].as<Vector3Df>();
                auto offset = object["offset"].as<Vector3Df>();
                auto meshData = loadModel(path, scale, rotate, offset);
                auto vertices = meshData.vertices;
                if (!vertices.empty()) {
                    TriangleMeshData meshData;
                    createTriangleMeshData(vertices, meshData);
                    triangleMeshData.push_back(meshData);
                    createMesh(spheres, i, meshData.triangleCount, meshData.triangleData, meshData.AABBData, meshData.boundsMin, meshData.boundsMax, materials[3]);
                }
            }
            break;
        }
    }

    //createCube(spheres, 1, centerAABB, extendAABB, materials[0]);
    //createSphere(spheres, 6, { -0.225f, -0.325f, -0.25f }, 0.175f, materials[6]);
    //createSphere(spheres, 7, { 0.275f, -0.325f, -0.125f }, 0.175f, materials[7]);
    //createSphere(spheres, 5, { 0.0f, -102.0f, -1.0f }, 100.0f, materials[4]);
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
    //clearBackBuffers<<<gridSize, blockSize>>>(canvas);
    //gpuErrorCheck(cudaDeviceSynchronize());
    canvas->clearAccumulationBuffer();
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
    renderKernel<<<gridSize, blockSize>>>(canvas, camera, randStates, nullptr, payload);
    gpuErrorCheck(cudaDeviceSynchronize());

    imageData->data = canvas->getPixelBuffer();
#else
    auto* counter = createObjectPtr<int32_t>();
    (*counter) = 0;
    canvas->incrementSampleCount();
    renderKernel<<<gridSize, blockSize>>>(canvas, camera, randStates, counter, payload);
    gpuErrorCheck(cudaDeviceSynchronize());

    deleteObject(counter);
#endif
}

void cleanup() {
    deleteObject(randStates);

    deleteDeviceObjectArray<<<1, 1>>>(spheres, OBJECTS);
    deleteDeviceObjectArray<<<1, 1>>>(materials, MATERIALS);

    gpuErrorCheck(cudaDeviceSynchronize());

    for (auto i = 0; i < MESHES; i++) {
        gpuErrorCheck(cudaDestroyTextureObject(triangleMeshData[i].triangleData));
        gpuErrorCheck(cudaDestroyTextureObject(triangleMeshData[i].AABBData));
        gpuErrorCheck(cudaFree(triangleMeshData[i].vertices));
        gpuErrorCheck(cudaFree(triangleMeshData[i].AABBs));
    }
    deleteObject(AABB);
    deleteObject(spheres);
    deleteObject(materials);

    deleteObject(camera);
    canvas->uninitialize();
    deleteObject(canvas);

    deleteObject(payload.cudaBVHindexesOrTrilists2);
    deleteObject(payload.cudaBVHlimits2);
    deleteObject(payload.cudaTriIdxList2);
    deleteObject(payload.cudaTriangles2);
    deleteObject(payload.cudaTriangleIntersectionData2);
    deleteObject(payload.cudaVertices2);
    cudaDestroyTextureObject(triangleDataTexture.triIdxListTexture);
    cudaDestroyTextureObject(triangleDataTexture.CFBVHlimitsTexture);
    cudaDestroyTextureObject(triangleDataTexture.CFBVHindexesOrTrilistsTexture);
    cudaDestroyTextureObject(triangleDataTexture.trianglesTexture);

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