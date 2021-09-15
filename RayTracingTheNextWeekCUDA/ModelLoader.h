#pragma once

#include <string>
#include <memory>
#include <vector>

#include "CUDATypes.h"
#include "LinearAlgebra.h"

struct MeshData {
    std::vector<Vector3Df> vertices;
    std::vector<Vector3Df> uniqueVertices;
    std::vector<uint32_t> indices;
};

std::shared_ptr<class Model> loadModel(const std::string& fileName, const std::string& inName, const std::string& materialPath, const std::string& texturePath);

MeshData loadModel(const std::string& fileName, const Vector3Df& scale = Vector3Df(1.0f, 1.0f, 1.0f),
                                                const Vector3Df& rotate = Vector3Df(0.0f, 0.0f, 0.0f),
                                                const Vector3Df& offset = Vector3Df(0.0f, 0.0f, 0.0f));

