#pragma once

#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_GLOBAL __global__
#define CUDA_CONSTANT __constant__

#include "vector_types.h"
#include "cutil_math.h"

#ifdef USE_DOUBLE
using Float = double;
using Float2 = double2;
using Float3 = double3;
using Float4 = double4;
#define make_float3 make_double3
#else
using Float = float;
using Float2 = float2;
using Float3 = float3;
using Float4 = float4;
#define fmod fmodf
#define sqrt sqrtf
#define cos cosf
#define sin sinf
#define tan tanf
#define floor floorf
#define rsqrt rsqrtf
#define pow powf
#endif