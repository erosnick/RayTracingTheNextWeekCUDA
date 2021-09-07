#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDA.h"

#include <cstdint>

class Vec3 {
public:
    CUDA_HOST_DEVICE Vec3() {
    }

    CUDA_HOST_DEVICE constexpr Vec3(Float element0, Float element1, Float element2)
    : elements({ element0, element1, element2 }) {
    }

    CUDA_HOST_DEVICE Vec3(const Float3& inElements) {
        elements = inElements;
    }

    CUDA_HOST_DEVICE Vec3 operator-() const {
        return -elements;
    }

    CUDA_HOST_DEVICE Float operator[](int32_t index) const {
        switch (index)
        {
        case 0:
            return elements.x;
            break;
        case 1:
            return elements.y;
            break;
        case 2:
            return elements.z;
            break;
        default:
            return -1.0f;
            break;
        }
    }

    CUDA_HOST_DEVICE Float& operator[](int32_t index) {
        return (*this)[index];
    }

    CUDA_HOST_DEVICE Vec3& operator+=(const Vec3& v) {
        elements.x += v.x();
        elements.y += v.y();
        elements.z += v.z();

        return *this;
    }

    CUDA_HOST_DEVICE Vec3& operator*=(Float t) {
        elements.x *= t;
        elements.y *= t;
        elements.z *= t;

        return *this;
    }

    CUDA_HOST_DEVICE Vec3& operator/=(Float t) {
        return *this *= 1.0f / t;
    }

    CUDA_HOST_DEVICE Float length() const {
        return sqrt(lengthSquared());
    }

    CUDA_HOST_DEVICE Float lengthSquared() const {
        return dot(elements, elements);
    }

    CUDA_HOST_DEVICE Float x() const {
        return elements.x;
    }

    CUDA_HOST_DEVICE Float y() const {
        return elements.y;
    }

    CUDA_HOST_DEVICE Float z() const {
        return elements.z;
    }

    Float3 elements;
};

inline CUDA_HOST_DEVICE Vec3 operator+(const Vec3& u, const Vec3& v) {
    return Vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

inline CUDA_HOST_DEVICE Vec3 operator+(const Vec3& u, Float v) {
    return Vec3(u.x() + v, u.y() + v, u.z() + v);
}

inline CUDA_HOST_DEVICE Vec3 operator+(Float v, const Vec3& u) {
    return u + v;
}

inline CUDA_HOST_DEVICE Vec3 operator-(const Vec3& u, const Vec3& v) {
    return Vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

inline CUDA_HOST_DEVICE Vec3 operator*(const Vec3& u, const Vec3& v) {
    return Vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

inline CUDA_HOST_DEVICE Vec3 operator*(Float t, const Vec3& v) {
    return Vec3(t * v.x(), t * v.y(), t * v.z());
}

inline CUDA_HOST_DEVICE Vec3 operator*(const Vec3& v, Float t) {
    return t * v;
}

inline CUDA_HOST_DEVICE Vec3 operator/(Vec3 v, Float t) {
    return (1 / t) * v;
}

inline CUDA_HOST_DEVICE Float dot(const Vec3& u, const Vec3& v) {
    return u.x() * v.x()
         + u.y() * v.y()
         + u.z() * v.z();
}

inline CUDA_HOST_DEVICE Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(u.y() * v.z() - u.z() * v.y(),
                u.z() * v.x() - u.x() * v.z(),
                u.x() * v.y() - u.y() * v.x());
}

inline CUDA_HOST_DEVICE Vec3 normalize(const Vec3& v) {
    return v / v.length();
}

inline CUDA_HOST_DEVICE Vec3 lerp(const Vec3& v0, const Vec3& v1, Float t) {
    return (1.0f - t) * v0 + t * v1;
}