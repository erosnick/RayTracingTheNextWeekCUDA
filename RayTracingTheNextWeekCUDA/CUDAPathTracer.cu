/*
*  CUDA based triangle mesh path tracer using BVH acceleration by Sam lapere, 2016
*  BVH implementation based on real-time CUDA ray tracer by Thanassis Tsiodras,
*  http://users.softlab.ntua.gr/~ttsiod/cudarenderer-BVH.html
*  Interactive camera with depth of field based on CUDA path tracer code
*  by Peter Kutz and Yining Karl Li, https://github.com/peterkutz/GPUPathTracer
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

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "device_launch_parameters.h"
#include "cutil_math.h"
#include <cuda_runtime.h>

#include "CUDAPathTracer.h"
#include "Ray.h"
#include "Types.h"

#define M_PI 3.1415926535897932384626422832795028841971f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define NUDGE_FACTOR     1e-3f  // epsilon
#define samps  1 // samples
#define BVH_STACK_SIZE 32
#define SCREEN_DIST (height*2)

int texturewidth = 0;
int textureheight = 0;
int total_number_of_triangles;

__device__ int depth = 0;

TriangleDataTexture triangleDataTexture;

Vertex* cudaVertices;
float* cudaTriangleIntersectionData;
int* cudaTriIdxList = nullptr;
float* cudaBVHlimits = nullptr;
int* cudaBVHindexesOrTrilists = nullptr;
Triangle* cudaTriangles = nullptr;

struct Sphere {

	float rad;								// radius 
	float3 pos, emi, col;					// position, emission, color 
	ReflectionType reflectionType;			// reflection type (DIFFuse, SPECular, REFRactive)

	__device__ float intersect(const Ray& r) const { // returns distance, 0 if nohit 

		// Ray/sphere intersection
		// Quadratic formula required to solve ax^2 + bx + c = 0 
		// Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
		float3 direction = r.direction;
		float3 op = pos - r.origin;  // 
		float t, epsilon = 0.01f;
		float b = dot(op, direction);
		float disc = b * b - dot(op, op) + rad * rad; // discriminant
		if (disc < 0) return 0; else disc = sqrtf(disc);
		return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
	}
};

__device__ Sphere spheres[] = {

	// sun
	{ 1.6, { 0.0f, 2.8, 0 }, { 6, 4, 2 }, { 0.f, 0.f, 0.f }, ReflectionType::DIFFUSE },  // 37, 34, 30  X: links rechts Y: op neer
	//{ 1600, { 3000.0f, 10, 6000 }, { 17, 14, 10 }, { 0.f, 0.f, 0.f }, DIFF },

	// horizon sun2
	//	{ 1560, { 3500.0f, 0, 7000 }, { 50, 25, 2.5 }, { 0.f, 0.f, 0.f }, DIFF },  //  150, 75, 7.5

	// sky
	//{ 10000, { 50.0f, 40.8f, -1060 }, { 0.1, 0.3, 0.55 }, { 0.175f, 0.175f, 0.25f }, DIFF }, // 0.0003, 0.01, 0.15, or brighter: 0.2, 0.3, 0.6
	{ 10000, { 50.0f, 40.8f, -1060 }, { 0.51, 0.51, 0.51 }, { 0.175f, 0.175f, 0.25f }, ReflectionType::DIFFUSE },

	// ground
	{ 100000, { 0.0f, -100001.1, 0 }, { 0, 0, 0 }, { 0.5f, 0.0f, 0.0f }, ReflectionType::COAT },
	{ 100000, { 0.0f, -100001.2, 0 }, { 0, 0, 0 }, { 0.3f, 0.3f, 0.3f }, ReflectionType::DIFFUSE }, // double shell to prevent light leaking

	// horizon brightener
	{ 110000, { 50.0f, -110048.5, 0 }, { 3.6, 2.0, 0.2 }, { 0.f, 0.f, 0.f }, ReflectionType::DIFFUSE },
	// mountains
	//{ 4e4, { 50.0f, -4e4 - 30, -3000 }, { 0, 0, 0 }, { 0.2f, 0.2f, 0.2f }, DIFF },
	// white Mirr
	{ 1.1, { 1.6, 0, 1.0 }, { 0, 0.0, 0 }, { 0.9f, .9f, 0.9f }, ReflectionType::SPECULAR }
	// Glass
	//{ 0.3, { 0.0f, -0.4, 4 }, { .0, 0., .0 }, { 0.9f, 0.9f, 0.9f }, DIFF },
	// Glass2
	//{ 22, { 87.0f, 22, 24 }, { 0, 0, 0 }, { 0.9f, 0.9f, 0.9f }, SPEC },
};

// Helper function, that checks whether a ray intersects a bounding box (BVH node)
__device__ bool RayIntersectsBox(const Vector3Df& originInWorldSpace, const Vector3Df& rayInWorldSpace, int boxIdx, TriangleDataTexture triangleDataTexture)
{
	// set Tnear = - infinity, Tfar = infinity
	//
	// For each pair of planes P associated with X, Y, and Z do:
	//     (example using X planes)
	//     if direction Xd = 0 then the ray is parallel to the X planes, so
	//         if origin Xo is not between the slabs ( Xo < Xl or Xo > Xh) then
	//             return false
	//     else, if the ray is not parallel to the plane then
	//     begin
	//         compute the intersection distance of the planes
	//         T1 = (Xl - Xo) / Xd
	//         T2 = (Xh - Xo) / Xd
	//         If T1 > T2 swap (T1, T2) /* since T1 intersection with near plane */
	//         If T1 > Tnear set Tnear =T1 /* want largest Tnear */
	//         If T2 < Tfar set Tfar="T2" /* want smallest Tfar */
	//         If Tnear > Tfar box is missed so
	//             return false
	//         If Tfar < 0 box is behind ray
	//             return false
	//     end
	// end of for loop

	float Tnear, Tfar;
	Tnear = -FLT_MAX;
	Tfar = FLT_MAX;

	float2 limits;

	// box intersection routine
#define CHECK_NEAR_AND_FAR_INTERSECTION(c)							    \
    if (rayInWorldSpace.##c == 0.f) {									\
	if (originInWorldSpace.##c < limits.x) return false;				\
	if (originInWorldSpace.##c > limits.y) return false;				\
	} else {															\
	float T1 = (limits.x - originInWorldSpace.##c)/rayInWorldSpace.##c;	\
	float T2 = (limits.y - originInWorldSpace.##c)/rayInWorldSpace.##c;	\
	if (T1>T2) { float tmp=T1; T1=T2; T2=tmp; }						    \
	if (T1 > Tnear) Tnear = T1;											\
	if (T2 < Tfar)  Tfar = T2;											\
	if (Tnear > Tfar)	return false;									\
	if (Tfar < 0.f)	return false;									    \
	}

	limits = tex1Dfetch<float2>(triangleDataTexture.CFBVHlimitsTexture, 3 * boxIdx); // box.bottom._x/top._x placed in limits.x/limits.y
	//limits = make_float2(cudaBVHlimits[6 * boxIdx + 0], cudaBVHlimits[6 * boxIdx + 1]);
	CHECK_NEAR_AND_FAR_INTERSECTION(x)
		limits = tex1Dfetch<float2>(triangleDataTexture.CFBVHlimitsTexture, 3 * boxIdx + 1); // box.bottom._y/top._y placed in limits.x/limits.y
		//limits = make_float2(cudaBVHlimits[6 * boxIdx + 2], cudaBVHlimits[6 * boxIdx + 3]);
	CHECK_NEAR_AND_FAR_INTERSECTION(y)
		limits = tex1Dfetch<float2>(triangleDataTexture.CFBVHlimitsTexture, 3 * boxIdx + 2); // box.bottom._z/top._z placed in limits.x/limits.y
		//limits = make_float2(cudaBVHlimits[6 * boxIdx + 4], cudaBVHlimits[6 * boxIdx + 5]);
	CHECK_NEAR_AND_FAR_INTERSECTION(z)

		// If Box survived all above tests, return true with intersection point Tnear and exit point Tfar.
		return true;
}

//////////////////////////////////////////
//	BVH intersection routine	//
//	using CUDA texture memory	//
//////////////////////////////////////////

// there are 3 forms of the BVH: a "pure" BVH, a cache-friendly BVH (taking up less memory space than the pure BVH)
// and a "textured" BVH which stores its data in CUDA texture memory (which is cached). The last one is gives the 
// best performance and is used here.

__device__ bool BVH_IntersectTriangles(
	int* cudaBVHindexesOrTrilists, const Vector3Df& origin, const Vector3Df& ray, unsigned avoidSelf,
	int& pBestTriIdx, Vector3Df& pointHitInWorldSpace, float& kAB, float& kBC, float& kCA, float& hitdist,
	float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList, Vector3Df& boxnormal, ReflectionType& reflectionType, TriangleDataTexture triangleDataTexture)
{
	// in the loop below, maintain the closest triangle and the point where we hit it:
	pBestTriIdx = -1;
	float bestTriDist;

	// start from infinity
	bestTriDist = FLT_MAX;

	// create a stack for each ray
	// the stack is just a fixed size array of indices to BVH nodes
	int stack[BVH_STACK_SIZE];

	int stackIdx = 0;
	stack[stackIdx++] = 0;
	Vector3Df hitpoint;

	// while the stack is not empty
	while (stackIdx) {

		// pop a BVH node (or AABB, Axis Aligned Bounding Box) from the stack
		int boxIdx = stack[stackIdx - 1];
		//uint* pCurrent = &cudaBVHindexesOrTrilists[boxIdx]; 

		// decrement the stackindex
		stackIdx--;

		// fetch the data (indices to childnodes or index in triangle list + trianglecount) associated with this node
		uint4 data = tex1Dfetch<uint4>(triangleDataTexture.CFBVHindexesOrTrilistsTexture, boxIdx);

		// original, "pure" BVH form...
		//if (!pCurrent->IsLeaf()) {

		// cache-friendly BVH form...
		//if (!(cudaBVHindexesOrTrilists[4 * boxIdx + 0] & 0x80000000)) { // INNER NODE

		// texture memory BVH form...

		// determine if BVH node is an inner node or a leaf node by checking the highest bit (bitwise AND operation)
		// inner node if highest bit is 1, leaf node if 0

		if (!(data.x & 0x80000000)) {   // INNER NODE

			// if ray intersects inner node, push indices of left and right child nodes on the stack
			if (RayIntersectsBox(origin, ray, boxIdx, triangleDataTexture)) {

				//stack[stackIdx++] = pCurrent->u.inner._idxRight;
				//stack[stackIdx++] = cudaBVHindexesOrTrilists[4 * boxIdx + 1];
				stack[stackIdx++] = data.y; // right child node index

				//stack[stackIdx++] = pCurrent->u.inner._idxLeft;
				//stack[stackIdx++] = cudaBVHindexesOrTrilists[4 * boxIdx + 2];
				stack[stackIdx++] = data.z; // left child node index

				// return if stack size is exceeded
				if (stackIdx > BVH_STACK_SIZE)
				{
					return false;
				}
			}
		}
		else { // LEAF NODE

			// original, "pure" BVH form...
			// BVHLeaf *p = dynamic_cast<BVHLeaf*>(pCurrent);
			// for(std::list<const Triangle*>::iterator it=p->_triangles.begin();
			//    it != p->_triangles.end();
			//    it++)

			// cache-friendly BVH form...
			// for(unsigned i=pCurrent->u.leaf._startIndexInTriIndexList;
			//    i<pCurrent->u.leaf._startIndexInTriIndexList + (pCurrent->u.leaf._count & 0x7fffffff);

			// texture memory BVH form...
			// for (unsigned i = cudaBVHindexesOrTrilists[4 * boxIdx + 3]; i< cudaBVHindexesOrTrilists[4 * boxIdx + 3] + (cudaBVHindexesOrTrilists[4 * boxIdx + 0] & 0x7fffffff); i++) { // data.w = number of triangles in leaf

			// loop over every triangle in the leaf node
			// data.w is start index in triangle list
			// data.x stores number of triangles in leafnode (the bitwise AND operation extracts the triangle number)
			auto dataGroupSize = 6;
			for (unsigned i = data.w; i < data.w + (data.x & 0x7fffffff); i++) {

				// original, "pure" BVH form...
				//const Triangle& triangle = *(*it);

				// cache-friendly BVH form...
				//const Triangle& triangle = pTriangles[cudaTriIdxList[i]];

				// texture memory BVH form...
				// fetch the index of the current triangle
				int idx = tex1Dfetch<uint1>(triangleDataTexture.triIdxListTexture, i).x;
				//int idx = cudaTriIdxList[i];

				// check if triangle is the same as the one intersected by previous ray
				// to avoid self-reflections/refractions
				if (avoidSelf == idx)
					continue;

				// fetch triangle center and normal from texture memory
				float4 center = tex1Dfetch<float4>(triangleDataTexture.trianglesTexture, dataGroupSize * idx);
				float4 normal = tex1Dfetch<float4>(triangleDataTexture.trianglesTexture, dataGroupSize * idx + 1);

				// use the pre-computed triangle intersection data: normal, d, e1/d1, e2/d2, e3/d3
				float k = dot(normal, ray);
				if (k == 0.0f)
					continue; // this triangle is parallel to the ray, ignore it.

				float s = (normal.w - dot(normal, origin)) / k;
				if (s <= 0.0f) // this triangle is "behind" the origin.
					continue;
				if (s <= NUDGE_FACTOR)  // epsilon
					continue;
				Vector3Df hit = ray * s;
				hit += origin;

				// ray triangle intersection
				// Is the intersection of the ray with the triangle's plane INSIDE the triangle?
				
				float4 ee1 = tex1Dfetch<float4>(triangleDataTexture.trianglesTexture, dataGroupSize * idx + 2);
				//float4 ee1 = make_float4(cudaTriangleIntersectionData[20 * idx + 8], cudaTriangleIntersectionData[20 * idx + 9], cudaTriangleIntersectionData[20 * idx + 10], cudaTriangleIntersectionData[20 * idx + 11]);
				float kt1 = dot(ee1, hit) - ee1.w;
				if (kt1 < 0.0f) continue;

				float4 ee2 = tex1Dfetch<float4>(triangleDataTexture.trianglesTexture, dataGroupSize * idx + 3);
				//float4 ee2 = make_float4(cudaTriangleIntersectionData[20 * idx + 12], cudaTriangleIntersectionData[20 * idx + 13], cudaTriangleIntersectionData[20 * idx + 14], cudaTriangleIntersectionData[20 * idx + 15]);
				float kt2 = dot(ee2, hit) - ee2.w;
				if (kt2 < 0.0f) continue;

				float4 ee3 = tex1Dfetch<float4>(triangleDataTexture.trianglesTexture, dataGroupSize * idx + 4);
				//float4 ee3 = make_float4(cudaTriangleIntersectionData[20 * idx + 16], cudaTriangleIntersectionData[20 * idx + 17], cudaTriangleIntersectionData[20 * idx + 18], cudaTriangleIntersectionData[20 * idx + 19]);
				float kt3 = dot(ee3, hit) - ee3.w;
				if (kt3 < 0.0f) continue;

				float4 extra = tex1Dfetch<float4>(triangleDataTexture.trianglesTexture, dataGroupSize * idx + 5);
				// ray intersects triangle, "hit" is the world space coordinate of the intersection.
				{
					// is this intersection closer than all the others?
					float hitZ = distancesq(origin, hit);
					if (hitZ < bestTriDist) {

						// maintain the closest hit
						bestTriDist = hitZ;
						hitdist = sqrtf(bestTriDist);
						pBestTriIdx = idx;
						pointHitInWorldSpace = hit;
						reflectionType = static_cast<ReflectionType>(extra.x);
						// store barycentric coordinates (for texturing, not used for now)
						kAB = kt1;
						kBC = kt2;
						kCA = kt3;
					}
				}
			}
		}
	}

	return pBestTriIdx != -1;
}

//////////////////////
// PATH TRACING
//////////////////////

__device__ Vector3Df pathTrace(curandState* randstate, Vector3Df originInWorldSpace, Vector3Df rayInWorldSpace, int avoidSelf, Triangle* pTriangles, int* cudaBVHindexesOrTrilists,
	float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList, TriangleDataTexture triangleDataTexture)
{
	// colour mask
	Vector3Df mask = Vector3Df(1.0f, 1.0f, 1.0f);
	// accumulated colour
	Vector3Df accucolor = Vector3Df(0.0f, 0.0f, 0.0f);

	for (int bounces = 0; bounces < 5; bounces++) {  // iteration up to 5 bounces (instead of recursion in CPU code)
		int sphere_id = -1;
		int triangle_id = -1;
		int pBestTriIdx = -1;
		int geomtype = -1;
		const Triangle* pBestTri = nullptr;
		Vector3Df pointHitInWorldSpace;
		float kAB = 0.f, kBC = 0.f, kCA = 0.f; // distances from the 3 edges of the triangle (from where we hit it), to be used for texturing

		float tmin = 1e20f;
		float tmax = -1e20f;
		float d = 1e20f;
		float scene_t = 1e20f;
		float inf = 1e20f;
		float hitdistance = 1e20;
		Vector3Df f = Vector3Df(0.0f, 0.0f, 0.0f);
		Vector3Df emit = Vector3Df(0.0f, 0.0f, 0.0f);
		Vector3Df x; // intersection point
		Vector3Df n; // normal
		Vector3Df nl; // oriented normal
		Vector3Df boxnormal = Vector3Df(0.0f, 0.0f, 0.0f);
		Vector3Df dw; // ray direction of next path segment
		ReflectionType reflectionType;

		float3 rayOrigin = make_float3(originInWorldSpace.x, originInWorldSpace.y, originInWorldSpace.z);
		float3 rayDirection = make_float3(rayInWorldSpace.x, rayInWorldSpace.y, rayInWorldSpace.z);

		// intersect all triangles in the scene stored in BVH
		BVH_IntersectTriangles(
			cudaBVHindexesOrTrilists, originInWorldSpace, rayInWorldSpace, avoidSelf,
			pBestTriIdx, pointHitInWorldSpace, kAB, kBC, kCA, hitdistance, cudaBVHlimits,
			cudaTriangleIntersectionData, cudaTriIdxList, boxnormal, reflectionType, triangleDataTexture);

		// intersect all spheres in the scene
		float numspheres = sizeof(spheres) / sizeof(Sphere);
		for (auto i = int(numspheres); i--;) {  // for all spheres in scene
			// keep track of distance from origin to closest intersection point
			if ((d = spheres[i].intersect(Ray(rayOrigin, rayDirection))) && d < scene_t) { scene_t = d; sphere_id = i; geomtype = 1; }
		}

		// set avoidSelf to current triangle index to avoid intersection between this triangle and the next ray, 
		// so that we don't get self-shadow or self-reflection from this triangle...
		avoidSelf = pBestTriIdx;

		if (hitdistance < scene_t && hitdistance > 0.002f) // EPSILON
		{
			scene_t = hitdistance;
			triangle_id = pBestTriIdx;
			geomtype = 2;
		}

		if (scene_t > 1e20f) return Vector3Df(0.0f, 0.0f, 0.0f);

		// SPHERES:
		if (geomtype == 1) {

			Sphere& sphere = spheres[sphere_id]; // hit object with closest intersection
			x = originInWorldSpace + rayInWorldSpace * scene_t;  // intersection point on object
			n = Vector3Df(x.x - sphere.pos.x, x.y - sphere.pos.y, x.z - sphere.pos.z);		// normal
			n.normalize();
			nl = dot(n, rayInWorldSpace) < 0 ? n : n * -1; // correctly oriented normal
			f = Vector3Df(sphere.col.x, sphere.col.y, sphere.col.z);   // object colour
			reflectionType = sphere.reflectionType;
			emit = Vector3Df(sphere.emi.x, sphere.emi.y, sphere.emi.z);  // object emission
			accucolor += (mask * emit);
		}

		// TRIANGLES:5
		if (geomtype == 2) {

			pBestTri = &pTriangles[triangle_id];

			x = pointHitInWorldSpace;  // intersection point
			n = pBestTri->_normal;  // normal

			n.normalize();
			nl = dot(n, rayInWorldSpace) < 0 ? n : n * -1;  // correctly oriented normal

			Vector3Df colour = Vector3Df(0.9f, 0.3f, 0.0f); // hardcoded triangle colour

			//reflectionType = ReflectionType::METAL;
			f = colour;
			emit = Vector3Df(0.0f, 0.0f, 0.0f);  // object emission
			accucolor += (mask * emit);
		}

		// basic material system, all parameters are hard-coded (such as phong exponent, index of refraction)

		// diffuse material, based on smallpt by Kevin Beason 
		if (reflectionType == ReflectionType::DIFFUSE) {

			// pick two random numbers
			float phi = 2 * M_PI * curand_uniform(randstate);
			float r2 = curand_uniform(randstate);
			float r2s = sqrtf(r2);

			// compute orthonormal coordinate frame uvw with hitpoint as origin 
			Vector3Df w = nl; w.normalize();
			Vector3Df u = cross((fabs(w.x) > .1 ? Vector3Df(0, 1, 0) : Vector3Df(1, 0, 0)), w); u.normalize();
			Vector3Df v = cross(w, u);

			// compute cosine weighted random ray direction on hemisphere 
			dw = u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2);
			dw.normalize();

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + w * 0.01;  // scene size dependent

			// multiply mask with colour of object
			mask *= f;
		}

		// Phong metal material from "Realistic Ray Tracing", P. Shirley
		if (reflectionType == ReflectionType::METAL) {

			// compute random perturbation of ideal reflection vector
			// the higher the phong exponent, the closer the perturbed vector is to the ideal reflection direction
			float phi = 2 * M_PI * curand_uniform(randstate);
			float r2 = curand_uniform(randstate);
			float phongexponent = 20;
			float cosTheta = powf(1 - r2, 1.0f / (phongexponent + 1));
			float sinTheta = sqrtf(1 - cosTheta * cosTheta);

			// create orthonormal basis uvw around reflection vector with hitpoint as origin 
			// w is ray direction for ideal reflection
			Vector3Df w = rayInWorldSpace - n * 2.0f * dot(n, rayInWorldSpace); w.normalize();
			Vector3Df u = cross((fabs(w.x) > .1 ? Vector3Df(0, 1, 0) : Vector3Df(1, 0, 0)), w); u.normalize();
			Vector3Df v = cross(w, u); // v is normalised by default

			// compute cosine weighted random ray direction on hemisphere 
			dw = u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta;
			dw.normalize();

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + w * 0.01;  // scene size dependent

			// multiply mask with colour of object
			mask *= f;
		}

		// specular material (perfect mirror)
		if (reflectionType == ReflectionType::SPECULAR) {

			// compute reflected ray direction according to Snell's law
			dw = rayInWorldSpace - n * 2.0f * dot(n, rayInWorldSpace);

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + nl * 0.01;   // scene size dependent

			// multiply mask with colour of object
			mask *= f;
		}

		// COAT material based on https://github.com/peterkutz/GPUPathTracer
		// randomly select diffuse or specular reflection
		// looks okay-ish but inaccurate (no Fresnel calculation yet)
		if (reflectionType == ReflectionType::COAT) {

			float rouletteRandomFloat = curand_uniform(randstate);
			float threshold = 0.05f;
			Vector3Df specularColor = Vector3Df(1, 1, 1);  // hard-coded
			bool reflectFromSurface = (rouletteRandomFloat < threshold); //computeFresnel(make_Vector3Df(n.x, n.y, n.z), incident, incidentIOR, transmittedIOR, reflectionDirection, transmissionDirection).reflectionCoefficient);

			if (reflectFromSurface) { // calculate perfectly specular reflection

				// Ray reflected from the surface. Trace a ray in the reflection direction.
				// TODO: Use Russian roulette instead of simple multipliers! (Selecting between diffuse sample and no sample (absorption) in this case.)

				mask *= specularColor;
				dw = rayInWorldSpace - n * 2.0f * dot(n, rayInWorldSpace);

				// offset origin next path segment to prevent self intersection
				pointHitInWorldSpace = x + nl * 0.01; // scene size dependent
			}

			else {  // calculate perfectly diffuse reflection

				float r1 = 2 * M_PI * curand_uniform(randstate);
				float r2 = curand_uniform(randstate);
				float r2s = sqrtf(r2);

				// compute orthonormal coordinate frame uvw with hitpoint as origin 
				Vector3Df w = nl; w.normalize();
				Vector3Df u = cross((fabs(w.x) > .1 ? Vector3Df(0, 1, 0) : Vector3Df(1, 0, 0)), w); u.normalize();
				Vector3Df v = cross(w, u);

				// compute cosine weighted random ray direction on hemisphere 
				dw = u * cosf(r1) * r2s + v * sinf(r1) * r2s + w * sqrtf(1 - r2);
				dw.normalize();

				// offset origin next path segment to prevent self intersection
				pointHitInWorldSpace = x + nl * 0.01;  // // scene size dependent

				// multiply mask with colour of object
				mask *= f;
				//mask *= make_Vector3Df(0.15f, 0.15f, 0.15f);  // gold metal
			}
		} // end COAT

		// perfectly refractive material (glass, water)
		if (reflectionType == ReflectionType::REFRACTION) {

			bool into = dot(n, nl) > 0; // is ray entering or leaving refractive material?
			float nc = 1.0f;  // Index of Refraction air
			float nt = 1.5f;  // Index of Refraction glass/water
			float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
			float ddn = dot(rayInWorldSpace, nl);
			float cos2t = 1.0f - nnt * nnt * (1.f - ddn * ddn);

			if (cos2t < 0.0f) // total internal reflection 
			{
				dw = rayInWorldSpace;
				dw -= n * 2.0f * dot(n, rayInWorldSpace);

				// offset origin next path segment to prevent self intersection
				pointHitInWorldSpace = x + nl * 0.01; // scene size dependent
			}
			else // cos2t > 0
			{
				// compute direction of transmission ray
				Vector3Df tdir = rayInWorldSpace * nnt;
				tdir -= n * ((into ? 1 : -1) * (ddn * nnt + sqrtf(cos2t)));
				tdir.normalize();

				float R0 = (nt - nc) * (nt - nc) / (nt + nc) * (nt + nc);
				float c = 1.f - (into ? -ddn : dot(tdir, n));
				float Re = R0 + (1.f - R0) * c * c * c * c * c;
				float Tr = 1 - Re; // Transmission
				float P = .25f + .5f * Re;
				float RP = Re / P;
				float TP = Tr / (1.f - P);

				// randomly choose reflection or transmission ray
				if (curand_uniform(randstate) < 0.25) // reflection ray
				{
					mask *= RP;
					dw = rayInWorldSpace;
					dw -= n * 2.0f * dot(n, rayInWorldSpace);

					pointHitInWorldSpace = x + nl * 0.01; // scene size dependent
				}
				else // transmission ray
				{
					mask *= TP;
					dw = tdir; //r = Ray(x, tdir); 
					pointHitInWorldSpace = x + nl * 0.001f; // epsilon must be small to avoid artefacts
				}
			}
		}

		// set up origin and direction of next path segment
		originInWorldSpace = pointHitInWorldSpace;
		rayInWorldSpace = dw;
	}

	return Vector3Df(accucolor.x, accucolor.y, accucolor.z);
}