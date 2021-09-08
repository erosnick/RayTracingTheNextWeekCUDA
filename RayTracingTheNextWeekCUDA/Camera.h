#pragma once

#include <cstdint>

#include "Ray.h"
#include "Constants.h"
#include "Utils.h"

class Camera {
public:
    CUDA_HOST_DEVICE Camera(const Float3& inEye, const Float3& inCenter, const Float3& inUp, Float inAspectRatio, Float inFOV = 90.0f, 
                            Float inAperture = 2.0f, Float inFocusDistance = 1.0f, Float inTime0 = 0.0f, Float inTime1 = 0.0f) {
        initialize(inEye, inCenter, inUp, inAspectRatio, inFOV, inAperture, inFocusDistance, inTime0, inTime1);
    }

    CUDA_HOST_DEVICE void initialize(const Float3& inEye, const Float3& inCenter, const Float3& inUp, Float inAspectRatio, Float inFOV = 90.0f, 
                                     Float inAperture = 2.0f, Float inFocusDistance = 1.0f, Float inTime0 = 0.0f, Float inTime1 = 0.0f) {
        eye = inEye;
        center = inCenter;
        up = inUp;
        aspectRatio = inAspectRatio;
        fov = inFOV;
        aperture = inAperture;
        focusDistance = inFocusDistance;
        movingSpeed = 3.0f;
        time0 = inTime0;
        time1 = inTime1;

        bIsDirty = true;
        updateViewMatrix();
    }

    ~Camera() {
        printf("I'm dead.\n");
    }

    void walk(Float delta) {
        eye += forward * movingSpeed * delta;
        center += forward * movingSpeed * delta;
        bIsDirty = true;
    }

    void strafe(Float delta) {
        eye += right * movingSpeed * delta;
        center += right * movingSpeed * delta;
        bIsDirty = true;
    }

    void raise(Float delta) {
        eye += up * movingSpeed * delta;
        center += up * movingSpeed * delta;
        bIsDirty = true;
    }

    void yaw(Float delta) {
        // Should rotate around up vector
        forward = normalize(rotateY(forward, delta));
        right = normalize(rotateY(right, delta));
        //up = glm::cross(right, forward);

        center = eye + forward;

        bIsDirty = true;
    }

    void pitch(Float delta) {
        // Should rotate around right vector
        forward = normalize(rotateX(forward, delta));
        //up = glm::normalize(rotation * up);
        center = eye + forward;

        bIsDirty = true;
    }

    CUDA_HOST_DEVICE inline void orbit(const Float3& target) {
        auto x = eye.x;
        auto z = eye.z;
        eye.x = (x - target.x) * cos(0.01f) - (z - target.z) * sin(0.01f) + target.x;
        eye.z = (x - target.x) * sin(0.01f) + (z - target.z) * cos(0.01f) + target.z;
        bIsDirty = true;
    }

    CUDA_HOST_DEVICE inline void updateViewMatrix() {
        if (bIsDirty) {
            scale = tan(Math::radians(fov / 2.0f));
            viewportHeight = 2.0f * scale;
            viewportWidth = viewportHeight * aspectRatio;

            forward = normalize(center - eye);
            right = normalize(cross(forward, up));
            trueUp = cross(right, forward);

            origin = eye;
            horizontal = viewportWidth * right * focusDistance;
            vertical = viewportHeight * trueUp * focusDistance;
            lowerLeftCorner = origin - horizontal / 2.0f - vertical / 2.0f + forward * focusDistance;

            updateLensRadius();
        }
    }

    CUDA_HOST_DEVICE inline void setAperture(Float inAperture) {
        aperture = inAperture;
        updateLensRadius();
    }

    CUDA_HOST_DEVICE inline void updateLensRadius() {
        bIsDirty = true;
        lensRadius = aperture / 2.0f;
    }
    
    CUDA_HOST_DEVICE inline const Float& getAperture() const {
        return aperture;
    }

    CUDA_HOST_DEVICE inline Float& getAperture() {
        return aperture;
    }

    CUDA_HOST_DEVICE inline const Float& getFOV() const {
        return fov;
    }

    CUDA_HOST_DEVICE inline Float& getFOV() {
        return fov;
    }

    CUDA_HOST_DEVICE inline const Float& getMovingSpeed() const {
        return movingSpeed;
    }

    CUDA_HOST_DEVICE inline Float& getMovingSpeed() {
        return movingSpeed;
    }

    inline void setDirty() {
        bIsDirty = true;
    }

    inline bool isDirty() const {
        return bIsDirty;
    }

    inline void resetDiryFlag() {
        bIsDirty = false;
    }

    CUDA_DEVICE inline Ray getRay(Float dx, Float dy, curandState* randState) {
        // Normally, all scene rays originate from the eye point.
        // In order to accomplish defocus blur, generate random 
        // scene rays originating from inside a disk centered at 
        // the eye point.The larger the radius, the greater the 
        // defocus blur.You can think of our original camera as 
        // having a defocus disk of radius zero(no blur at all), 
        // so all rays originated at the disk center(eye).
        auto random = lensRadius * Utils::randomInUnitDisk(randState);

        // Offset from lens center
        auto offset = right * random.x + trueUp * random.y;

        auto newOrigin = origin + offset;
        auto direction = lowerLeftCorner + dx * horizontal + dy * vertical - newOrigin;
        return Ray(origin + offset, normalize(direction), Utils::random(randState, time0, time1));
    }
private:
    Float aspectRatio;
    Float fov;
    Float scale;
    Float viewportHeight;
    Float viewportWidth;
    Float movingSpeed;
    Float aperture;
    Float focusDistance;
    Float lensRadius;

    // Shutter open/close time
    Float time0;
    Float time1;

    Float3 eye;
    Float3 center;
    Float3 forward;
    Float3 right;
    Float3 up;
    Float3 trueUp;
    Float3 horizontal;
    Float3 vertical;
    Float3 origin;
    Float3 lowerLeftCorner;

    bool bIsDirty;
};