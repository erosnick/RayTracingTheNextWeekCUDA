/*
*  CUDA based triangle mesh path tracer using BVH acceleration by Sam lapere, 2016
*  BVH implementation based on real-time CUDA ray tracer by Thanassis Tsiodras,
*  http://users.softlab.ntua.gr/~ttsiod/cudarenderer-BVH.html
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#ifndef __LINEAR_ALGEBRA_H_
#define __LINEAR_ALGEBRA_H_

#include <cuda_runtime.h> // for __host__  __device__
#include <cmath>

struct Vector3Df
{
	union {
		struct { float x, y, z; };
		float _v[3];
	};

	__host__ __device__ Vector3Df(float _x = 0.0f, float _y = 0.0, float _z = 0.0f) : x(_x), y(_y), z(_z) {}
	__host__ __device__ Vector3Df(const Vector3Df& v) : x(v.x), y(v.y), z(v.z) {}
	__host__ __device__ Vector3Df(const float3& v) : x(v.x), y(v.y), z(v.z) {}
	__host__ __device__ Vector3Df(const float4& v) : x(v.x), y(v.y), z(v.z) {}
	inline __host__ __device__ float length(){ return sqrtf(x*x + y*y + z*z); }
	// sometimes we dont need the sqrt, we are just comparing one length with another
	inline __host__ __device__ float lengthsq(){ return x*x + y*y + z*z; }
	inline __host__ __device__ void normalize(){ float norm = sqrtf(x*x + y*y + z*z); x /= norm; y /= norm; z /= norm; }
	inline __host__ __device__ Vector3Df& operator+=(const Vector3Df& v){ x += v.x; y += v.y; z += v.z; return *this; }
	inline __host__ __device__ Vector3Df& operator-=(const Vector3Df& v){ x -= v.x; y -= v.y; z -= v.z; return *this; }
	inline __host__ __device__ Vector3Df& operator*=(const float& a){ x *= a; y *= a; z *= a; return *this; }
	inline __host__ __device__ Vector3Df& operator*=(const Vector3Df& v){ x *= v.x; y *= v.y; z *= v.z; return *this; }
	inline __host__ __device__ Vector3Df operator*(float a) const{ return Vector3Df(x*a, y*a, z*a); }
	inline __host__ __device__ Vector3Df operator/(float a) const{ return Vector3Df(x/a, y/a, z/a); }
	inline __host__ __device__ Vector3Df operator*(const Vector3Df& v) const{ return Vector3Df(x * v.x, y * v.y, z * v.z); }
	inline __host__ __device__ Vector3Df operator+(const Vector3Df& v) const{ return Vector3Df(x + v.x, y + v.y, z + v.z); }
	inline __host__ __device__ Vector3Df operator+(const float& a) const { return Vector3Df(x + a, y + a, z + a); }
	inline __host__ __device__ Vector3Df operator-(const Vector3Df& v) const{ return Vector3Df(x - v.x, y - v.y, z - v.z); }
	inline __host__ __device__ Vector3Df operator-() const { return Vector3Df(-x, -y, -z); }
	inline __host__ __device__ Vector3Df& operator/=(const float& a){ x /= a; y /= a; z /= a; return *this; }
	inline __host__ __device__ bool operator!=(const Vector3Df& v){ return x != v.x || y != v.y || z != v.z; }

	inline __host__ __device__ operator float3() const { return make_float3(x, y, z); }
};

inline __host__ __device__ Vector3Df min3(const Vector3Df& v1, const Vector3Df& v2){ return Vector3Df(v1.x < v2.x ? v1.x : v2.x, v1.y < v2.y ? v1.y : v2.y, v1.z < v2.z ? v1.z : v2.z); }
inline __host__ __device__ Vector3Df max3(const Vector3Df& v1, const Vector3Df& v2){ return Vector3Df(v1.x > v2.x ? v1.x : v2.x, v1.y > v2.y ? v1.y : v2.y, v1.z > v2.z ? v1.z : v2.z); }
inline __host__ __device__ Vector3Df cross(const Vector3Df& v1, const Vector3Df& v2){ return Vector3Df(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x); }
inline __host__ __device__ float dot(const Vector3Df& v1, const Vector3Df& v2){ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
inline __host__ __device__ float dot(const Vector3Df& v1, const float4& v2){ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
inline __host__ __device__ float dot(const float4& v1, const Vector3Df& v2){ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
inline __host__ __device__ float distancesq(const Vector3Df& v1, const Vector3Df& v2){ return (v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z); }
inline __host__ __device__ float distance(const Vector3Df& v1, const Vector3Df& v2){ return sqrtf((v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z)); }

inline __host__ __device__ Vector3Df operator/(float a, const Vector3Df& b) {
	return Vector3Df(a / b.x, a / b.y, a / b.z);
}

inline __host__ __device__ Vector3Df operator*(float a, const Vector3Df& b) {
	return b * a;
}

inline __host__ __device__ Vector3Df operator+(const float& a, const Vector3Df& b) { 
	return b + a;
}

inline __host__ __device__ Vector3Df normalize(const Vector3Df& v) {
	float norm = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	return Vector3Df(v.x / norm, v.y / norm, v.z / norm);
}

inline __device__ __host__ Vector3Df rotateY(Vector3Df v, float radian) {
    return Vector3Df(v.x * cos(radian) + v.z * sin(radian), v.y,
        -v.x * sin(radian) + v.z * cos(radian));
}

inline __device__ __host__ Vector3Df rotateX(const Vector3Df& v, float radian) {
    return Vector3Df(v.x, v.y * cos(radian) - v.z * sin(radian),
					 v.y * sin(radian) + v.z * cos(radian));
}

inline __device__ __host__ Vector3Df rotate(const Vector3Df& v, const Vector3Df& axis, float radian) {
	auto cosAngle = cos(radian);
	auto sinAngle = sin(radian);
	float n1 = axis.x;
	float n2 = axis.y;
	float n3 = axis.z;
	float4 row0 = { n1 * n1 * (1.0f - cosAngle) + cosAngle, n1 * n2 * (1.0f - cosAngle) - n3 * sinAngle, n1 * n3 * (1.0f - cosAngle) + n2 * sinAngle, 0.0f };
	float4 row1 = { n1 * n2 * (1.0f - cosAngle) + n3 * sinAngle, n2 * n2 * (1.0f - cosAngle) + cosAngle, n2 * n3 * (1.0f - cosAngle) - n1 * sinAngle, 0.0f };
	float4 row2 = { n1 * n3 * (1.0f - cosAngle) - n2 * sinAngle, n2 * n3 * (1.0f - cosAngle) + n1 * sinAngle, n3 * n3 * (1.0f - cosAngle) + cosAngle, 0.0f };
	
	auto x = row0.x * v.x + row0.y * v.y + row0.z * v.z;
	auto y = row1.x * v.x + row1.y * v.y + row1.z * v.z;
	auto z = row2.x * v.x + row2.y * v.y + row2.z * v.z;

	return Vector3Df(x, y, z);
}

inline __device__ __host__ Vector3Df lerp(const Vector3Df& v0, const Vector3Df& v1, float t) {
    return (1.0f - t) * v0 + t * v1;
}

#endif
