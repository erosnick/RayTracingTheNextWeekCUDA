#ifndef __LOADER_H_
#define __LOADER_H_

#include "Types.h"

void panic(const char* fmt, ...);
void loadObject(const std::string& filename, ReflectionType reflectionType, int32_t meshIndex);
float processTriangleData(const Vector3Df& offset);
#endif
