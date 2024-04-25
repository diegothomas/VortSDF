#ifndef __CUDA_UTILITIES_H
#define __CUDA_UTILITIES_H

#include <stdio.h>
#include <iostream>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <algorithm>

#define THREAD_SIZE_X 16
#define THREAD_SIZE_Y 8
#define THREAD_SIZE_Z 8
#define PI_CU 3.141592653589793238462643383279502884197

using namespace std;

#define divUp(x,y) (x%y) ? ((x+y-1)/y) : (x/y)

/*float3 cross_prod(float3 a, float3 b) {
	return make_float3(a.y*b.z - a.z*b.y,
						-a.x*b.z + a.z*b.x,
						a.x*b.y - a.y*b.x);
}*/


inline __device__ void get_normal_f(float s1[3], float s2[3], float s3[3], float n[3]) {
    float a[3] = { s2[0] - s1[0], s2[1] - s1[1], s2[2] - s1[2] };
    float b[3] = { s3[0] - s1[0], s3[1] - s1[1], s3[2] - s1[2] };

    n[0] = a[1] * b[2] - a[2] * b[1];
    n[1] = -(a[0] * b[2] - a[2] * b[0]);
    n[2] = a[0] * b[1] - a[1] * b[0];
}

inline __device__ float squared_length_f(float a[3]) {
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

inline __device__ void cross_prod_d(double a[3], double b[3], double res[3]) {
    res[0] = a[1] * b[2] - a[2] * b[1];
    res[1] = -(a[0] * b[2] - a[2] * b[0]);
    res[2] = a[0] * b[1] - a[1] * b[0];
}

inline __device__ void cross_prod_f(float a[3], float b[3], float res[3]) {
    res[0] = a[1] * b[2] - a[2] * b[1];
    res[1] = -(a[0] * b[2] - a[2] * b[0]);
    res[2] = a[0] * b[1] - a[1] * b[0];
}

inline __device__ float dot_prod_d(double a[3], double b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline __device__ float dot_prod_f(float a[3], float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}


inline __device__ float4 make_float4(float3 a, float b) {
    return make_float4(a.x, a.y, a.z, b);
}

inline __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __device__ float3 operator*(float s, float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __device__ float4 operator*(float s, float4 a) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

inline __device__ float3 operator/(float3 a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

inline __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}


inline __device__ float3 operator-(float3 a, float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}

inline __device__ void operator+=(float3 a, float3 b) {
    a.x = a.x + b.x;
    a.y = a.y + b.y;
    a.z = a.z + b.z;
}

inline __device__ float elem(float3 a, int b) {
    if (b == 0)
        return a.x;
    if (b == 1)
        return a.y;
    if (b == 2)
        return a.z;
}

inline __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, 
                        -a.x * b.z + a.z * b.x,
                        a.x * b.y - a.y * b.x);
}

inline __device__ float norm_2(float3 a) {
    return sqrt(a.x * a.x + a.y* a.y + a.z* a.z);
}


struct bool3
{
    bool x, y, z;
};

inline  __device__ bool3 make_bool3(bool x, bool y, bool z)
{
    bool3 t; t.x = x; t.y = y; t.z = z; return t;
}


inline __device__ bool3 greaterThan(float3 a, float3 b) {
    return make_bool3(a.x > b.x, a.y > b.y, a.z > b.z);
}

inline __device__ float3 abs(float3 a) {
    return make_float3(abs(a.x), abs(a.y), abs(a.z));
}


inline __device__ float sign(float a) {
    if (a == 0.0f)
        return 0.0f;
    
    return a / abs(a);
}

inline __device__ float3 sign(float3 a) {
    return make_float3(sign(a.x), sign(a.y), sign(a.z));
}


inline __device__ float3 mix(float3 a, float3 b, bool3 t) {
    float3 res = make_float3(0.0f, 0.0f, 0.0f);
    res.x = t.x ? b.x : a.x;
    res.y = t.y ? b.y : a.y;
    res.z = t.z ? b.z : a.z;
    return res;
}




inline __device__ double3 operator-(double3 a, double3 b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ double3 operator/(double3 a, double s) {
    return make_double3(a.x / s, a.y / s, a.z / s);
}

inline __device__ double3 operator*(double3 a, double s) {
    return make_double3(a.x * s, a.y * s, a.z * s);
}

inline __device__ double3 operator-(double3 a) {
    return make_double3(-a.x, -a.y, -a.z);
}

inline __device__ double3 operator+(double3 a, double3 b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ double dot(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ double3 cross(double3 a, double3 b) {
    return make_double3(a.y * b.z - a.z * b.y,
        -a.x * b.z + a.z * b.x,
        a.x * b.y - a.y * b.x);
}



#endif