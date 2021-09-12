#pragma once

#include <string>
#include <memory>
#include <vector>

#include "CUDATypes.h"

std::shared_ptr<class Model> loadModel(const std::string& fileName, const std::string& inName, const std::string& materialPath, const std::string& texturePath);

std::vector<Float3> loadModel(const std::string& fileName);

