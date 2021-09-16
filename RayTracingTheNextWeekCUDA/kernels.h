#pragma once

#include "CUDATypes.h"
#include "Material.h"
#include "Canvas.h"
#include "Hitable.h"
#include "Sphere.h"
#include "Plane.h"
#include "TriangleMesh.h"
#include "Cube.h"

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
        delete* (object + i);
    }
}

CUDA_GLOBAL void createLambertianMaterialKernel(Material** material, int32_t index, Vector3Df albedo, Float absorb = 1.0f) {
    *(material + index) = new Lambertian(index, albedo, absorb);
}

void createLambertianMaterial(Material** material, int32_t index, Vector3Df albedo, Float absorb = 1.0f) {
    createLambertianMaterialKernel<<<1, 1>>>(material, index, albedo, absorb);
    gpuErrorCheck(cudaDeviceSynchronize());
}

CUDA_GLOBAL void createEmissionMaterialKernel(Material** material, int32_t index, Vector3Df albedo, Float intensity = 1.0f) {
    *(material + index) = new Emission(index, albedo, intensity);
}

void createEmissionMaterial(Material** material, int32_t index, Vector3Df albedo, Float intensity = 1.0f) {
    createEmissionMaterialKernel<<<1, 1>>>(material, index, albedo, intensity);
    gpuErrorCheck(cudaDeviceSynchronize());
}

CUDA_GLOBAL void createMetalMaterialKernel(Material** material, int32_t index, Vector3Df albedo, Float fuzz = 1.0f) {
    *(material + index) = new Metal(index, albedo, fuzz);
}

void createMetalMaterial(Material** material, int32_t index, Vector3Df albedo, Float fuzz = 1.0f) {
    createMetalMaterialKernel<<<1, 1>>>(material, index, albedo, fuzz);
    gpuErrorCheck(cudaDeviceSynchronize());
}

CUDA_GLOBAL void createDieletricMaterialKernel(Material** material, int32_t index, Float indexOfRefraction = 1.5f) {
    *(material + index) = new Dieletric(index, indexOfRefraction);
}

void createDieletricMaterial(Material** material, int32_t index, Float indexOfRefraction = 1.5f) {
    createDieletricMaterialKernel<<<1, 1>>>(material, index, indexOfRefraction);
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

CUDA_GLOBAL void createSphereKernel(Hitable** sphere, int32_t index, Vector3Df center, Float radius, Material* material, bool bShading) {
    *(sphere + index) = new Sphere(center, radius, material, bShading);
}

void createSphere(Hitable** sphere, int32_t index, Vector3Df center, Float radius, Material* material, bool bShading = true) {
    createSphereKernel<<<1, 1>>>(sphere, index, center, radius, material, bShading);
    gpuErrorCheck(cudaDeviceSynchronize());
}

CUDA_GLOBAL void createMovingSphereKernel(Hitable** sphere, int32_t index, Vector3Df center0, Vector3Df center1, Float time0, Float time1, Float radius, Material* material) {
    *(sphere + index) = new MovingSphere(center0, center1, time0, time1, radius, material);
}

void createMovingSphere(Hitable** sphere, int32_t index, Vector3Df center0, Vector3Df center1, Float time0, Float time1, Float radius, Material* material) {
    createMovingSphereKernel<<<1, 1>>>(sphere, index, center0, center1, time0, time1, radius, material);
    gpuErrorCheck(cudaDeviceSynchronize());
}

CUDA_GLOBAL void createPlaneKernel(Hitable** plane, int32_t index, Vector3Df position, Vector3Df normal, Vector3Df extend, Material* material, PlaneOrientation orientation, bool bTwoSide) {
    *(plane + index) = new Plane(position, normal, extend, material, orientation, bTwoSide);
}

void createPlane(Hitable** plane, int32_t index, const Vector3Df& position, const Vector3Df& normal, const Vector3Df& extend, Material* material, PlaneOrientation orientation, bool bTwoSide = true) {
    createPlaneKernel<<<1, 1>>>(plane, index, position, normal, extend, material, orientation, bTwoSide);
    gpuErrorCheck(cudaDeviceSynchronize());
}

CUDA_GLOBAL void createMeshKernel(Hitable** mesh, int32_t index, int32_t triangleCount, cudaTextureObject_t triangleData, cudaTextureObject_t AABBData, Vector3Df boundsMin, Vector3Df boundsMax, Material* material) {
    *(mesh + index) = new TriangleMesh(triangleCount, triangleData, AABBData, boundsMin, boundsMax, material);
}

void createMesh(Hitable** triangle, int32_t index, int32_t triangleCount, cudaTextureObject_t triangleData, cudaTextureObject_t AABBData, Vector3Df& boundsMin, const Vector3Df& boundsMax, Material* material) {
    createMeshKernel<<<1, 1>>>(triangle, index, triangleCount, triangleData, AABBData, boundsMin, boundsMax, material);
    gpuErrorCheck(cudaDeviceSynchronize());
}

CUDA_GLOBAL void createCubeKernel(Hitable** cube, int32_t index, Vector3Df position, Hitable** faces, Material* material) {
    *(cube + index) = new Cube(position, faces, material);
}

void createCube(Hitable** cube, int32_t index, const Vector3Df& center, const Vector3Df& extend, Material* material) {
    auto** faces = createObjectPtrArray<Hitable*>(6);

    createPlane(faces, 0, { center.x - extend.x,  center.y,  center.z }, { -1.0f,  0.0f,  0.0f }, extend, material, PlaneOrientation::YZ);   // Left
    createPlane(faces, 1, { center.x + extend.x,  center.y,  center.z }, {  1.0f,  0.0f,  0.0f }, extend, material, PlaneOrientation::YZ);   // Right
    createPlane(faces, 2, { center.x,  center.y + extend.y,  center.z }, {  0.0f,  1.0f,  0.0f }, extend, material, PlaneOrientation::XZ);   // Top
    createPlane(faces, 3, { center.x,  center.y - extend.y,  center.z }, {  0.0f, -1.0f,  0.0f }, extend, material, PlaneOrientation::XZ);   // Bottom
    createPlane(faces, 4, { center.x,  center.y,  center.z + extend.z }, {  0.0f,  0.0f,  1.0f }, extend, material, PlaneOrientation::XY);   // Front
    createPlane(faces, 5, { center.x,  center.y,  center.z - extend.z }, {  0.0f,  0.0f, -1.0f }, extend, material, PlaneOrientation::XY);   // Back
    gpuErrorCheck(cudaDeviceSynchronize());

    createCubeKernel<<<1, 1>>>(cube, index, center, faces, material);
    gpuErrorCheck(cudaDeviceSynchronize());

    gpuErrorCheck(cudaFree(faces));
}