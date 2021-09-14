#pragma once

#include <string>
#include <memory>
#include <vector>

#include "CUDATypes.h"

struct MeshData {
    std::vector<Float3> vertices;
    std::vector<Float3> uniqueVertices;
    std::vector<uint32_t> indices;
};

std::shared_ptr<class Model> loadModel(const std::string& fileName, const std::string& inName, const std::string& materialPath, const std::string& texturePath);

MeshData loadModel(const std::string& fileName, const Float3& scale = make_float3(1.0f, 1.0f, 1.0f), 
                                                const Float3& rotate = make_float3(0.0f, 0.0f, 0.0f), 
                                                const Float3& offset = make_float3(0.0f, 0.0f, 0.0f));

