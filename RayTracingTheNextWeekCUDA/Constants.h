#pragma once

#include "Vec3.h"
#include <cstdint>
#include <limits>

constexpr int32_t MAXELEMENTS = 8;

namespace Math {
    constexpr Float pi = 3.1415926535897932f;
    constexpr Float pi_3 = 1.0471975511965976f;
    constexpr Float pi_2 = 1.5707963267948966f;
    constexpr Float pi_4 = 0.7853981633974483f;
    constexpr Float pi_6 = 0.5235987755982988f;
    constexpr Float sqrt_2 = 1.414214f;
    constexpr Float sqrt_3 = 1.732051f;
    constexpr Float cos30d = 0.866025f;
    constexpr Float cos45d = 0.707107f;
    constexpr Float cos60d = 0.5f;
    constexpr Float sin30d = 0.5f;
    constexpr Float sin45d = 0.707107f;
    constexpr Float sin60D = 0.866025f;
    constexpr Float epsilon = 0.001f;
    constexpr Vec3 xAxis = Vec3(1.0f, 0.0f,  0.0f);
    constexpr Vec3 yAxis = Vec3(0.0f, 1.0f,  0.0f);
    constexpr Vec3 zAxis = Vec3(0.0f, 0.0f, -1.0f);
    constexpr Float infinity = std::numeric_limits<Float>::infinity();

    inline CUDA_HOST_DEVICE Float radians(Float degree) {
        return pi / 180.0f * degree;
    }

    inline CUDA_HOST_DEVICE Float degrees(Float radian) {
        return radian * 180.0f / pi;
    }
}

namespace Color {
    constexpr Float oneOver255 = 1.0f / 255;
    constexpr Vec3 black = Vec3(0.0f, 0.0f, 0.0f);
    constexpr Vec3 dawn = Vec3(0.1f, 0.1f, 0.1f);
    constexpr Vec3 white = Vec3(1.0f, 1.0f, 1.0f);
    constexpr Vec3 grey = Vec3(0.5f, 0.5f, 0.5f);
    constexpr Vec3 gray = Vec3(0.7f, 0.7f, 0.7f);
    constexpr Vec3 red = Vec3(1.0f, 0.0f, 0.0f);
    constexpr Vec3 green = Vec3(0.0f, 1.0f, 0.0f);
    constexpr Vec3 yellow = Vec3(1.0f, 1.0f, 0.0f);
    constexpr Vec3 purple = Vec3(1.0f, 0.0f, 1.0f);
    constexpr Vec3 blue = Vec3(0.0f, 0.0f, 1.0f);
    constexpr Vec3 pink = Vec3(1.0f, 0.55f, 0.55f);
    constexpr Vec3 skyBlue = Vec3(134, 203, 237);
    constexpr Vec3 moonstone = Vec3(60, 162, 200);
    constexpr Vec3 turquoise = Vec3(64, 224, 208);
    constexpr Vec3 limeGreen = Vec3(110, 198, 175);
    constexpr Vec3 roseRed = Vec3(0.76f, 0.12f, 0.34f);
    constexpr Vec3 crimsonRed = Vec3(0.86f, 0.08f, 0.24f);
    constexpr Vec3 lightGreen = Vec3(0.38f, 1.0f, 0.18f);
    constexpr Vec3 orange = Vec3(0.85f, 0.49f, 0.32f);
    constexpr Vec3 cornflower = Vec3(0.4f, 0.6f, 0.9f);
    constexpr Vec3 background = Vec3(0.235294f, 0.67451f, 0.843137f);
    constexpr Vec3 lightCornflower = Vec3(0.5f, 0.7f, 1.0f);
    constexpr Vec3 White() { return white; }
    constexpr Vec3 LightCornflower() { return lightCornflower; }
}