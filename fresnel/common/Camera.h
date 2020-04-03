// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "VectorMath.h"

#include "Random123/philox.h"
#include "uniform.hpp"

#ifndef __CAMERA_H__
#define __CAMERA_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#undef DEVICE
#ifdef __CUDACC__
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

namespace fresnel {

/** Camera models.

    Used to select the camera model in Camera.
*/
enum class CameraModel
    {
    orthographic, pinhole, thin_lens
    };

/** Store user provided camera properties.

    Camera uses these properties to compute quantities it needs when generating the camera rays.

    The user selects a camera model with the model field. The parameters used to define the camera
    depend on the model.

    All camera models use:

    * position: The camera position (the center of projection)
    * look_at: The point the camera looks at. This also sets the distance to the focal plane.
    * up: A vector pointing up.
    * h: The height of the image plane.

    Perspective and thin lens models also use:

    * f: The focal length of modelled lens.

    And the thin lens model uses:

    * f_stop: Set the aperture diameter to `f/f_stop`.

    All convenience methods and derived properties are implemented in Python.
*/
struct UserCamera
{
    vec3<float> position;
    vec3<float> look_at;
    vec3<float> up;
    float h;
    float f;
    float f_stop;
    CameraModel model;
};

/** Store an orthonormal basis u,v,w.

    The basis vectors are:

    * u: Points right
    * v: Points up
    * w: Points out of the screen (the camera looks in the -w direction)

    These are computed from position, look_at, and up from UserCamera.
*/
struct CameraBasis
{
    CameraBasis() {}
    explicit CameraBasis(const UserCamera& user) : v(user.up)
    {
        vec3<float> d = user.look_at - user.position;

        // normalize inputs
        d *= 1.0f / sqrtf(dot(d, d));
        v *= 1.0f / sqrtf(dot(v, v));

        // form right vector
        u = cross(d, v);
        u *= 1.0f / sqrtf(dot(u, u));

        // make set orthonormal by recomputing the up direction
        v = cross(u, d);
        v *= 1.0f / sqrtf(dot(v, v));

        // w points opposite the direction the camera faces
        w = -d;
    }

    vec3<float> u, v, w;
};

/** Implenatation of the camera models.

    RayGen generates image sample locations in fractional coordinates that range from -0.5 to 0.5 in
    the y direction and from -0.5*aspect to 0.5*aspect in the x direction, where aspect is the
    aspect ratio of the image. Given a sample location on the image plane, Camera generates a ray
    (origin and normalized direction) that should be sampled.

    Construct a Camera from a UserCamera. UserCamera provides all of the properties needed to define
    the camera model. Camera preprocesses some of these inputs into values usable when generating
    rays.

    Some camera models (such as thin lens) utilize random sampling. The generate() method takes a
    pixel location, a user random number seed value, and a sample index.

    TODO: refactor AA sampling from RayGen into the Camera class.
*/
class Camera
{
    public:
    Camera() {}

    /** Construct from a UserCamera.

        @param user Camera parameters.
        @param width Width of the image in pixels.
        @param height Height of the image in pixels.
        @param seed Random number seed.
    */
    explicit Camera(const UserCamera& user,
                    unsigned int width,
                    unsigned int height,
                    unsigned int seed)
        : m_p(user.position), m_basis(user), m_a(user.f/user.f_stop),
        m_model(user.model), m_width(width), m_height(height),m_seed(seed)
        {
        vec3<float> direction = user.look_at - user.position;
        m_focal_d = sqrtf(dot(direction, direction));

        // precompute focal plane height
        if (m_model == CameraModel::orthographic)
            {
            m_focal_h = user.h;
            }
        else
            {
            m_focal_h = user.h / user.f * m_focal_d;
            }
        }

    /** Generate a ray into the scene.

        @param origin [out] Origin of the generated ray.
        @param direction [out] Normalized direction of the generated ray.
        @param s Sample location in fractional image space
        @param i Pixel index in the x direction
        @param j Pixel index in the y direction
        @param sample Index of the sample to generate
    */
    DEVICE void generateRay(vec3<float> &origin,
                            vec3<float> &direction,
                            const vec2<float>& s,
                            unsigned int i,
                            unsigned int j,
                            unsigned int sample) const
    {
        if (m_model == CameraModel::orthographic)
            {
            origin = m_p + (s.y * m_basis.v + s.x * m_basis.u) * m_focal_h;
            direction = -m_basis.w;
            return;
            }

        // compute the central focal point
        vec3<float> F = m_p - m_basis.w * m_focal_d;
        // compute the point of convergence for rays passing through this pixel
        vec3<float> C = F + (s.y * m_basis.v + s.x * m_basis.u) * m_focal_h;

        if (m_model == CameraModel::pinhole)
        {
            origin = m_p;
        }
        else if (m_model == CameraModel::thin_lens)
        {
            // create the philox unique key for this RNG which includes the pixel ID and the random seed
            unsigned int pixel = j * m_width + i;
            r123::Philox4x32::ukey_type rng_key = {{pixel, m_seed}};

            // generate a random point in a circle
            r123::Philox4x32 rng;
            r123::Philox4x32::ctr_type rng_counter = {{0, 0, sample, rng_val_aperture}};
            r123::Philox4x32::ctr_type rng_u = rng(rng_counter, rng_key);
            float theta = r123::u01<float>(rng_u[0]) * 2.0f * float(M_PI);
            float r = r123::u01<float>(rng_u[1]) * m_a * 0.5f;

            vec3<float> offset = r * cosf(theta) * m_basis.u + r * sinf(theta) * m_basis.v;
            origin = m_p + offset;
        }

        direction = C - origin;
        direction *= 1.0f / sqrtf(dot(direction, direction));
    }

    /// Get the camera basis.
    const CameraBasis& getBasis() const
        {
        return m_basis;
        }


    private:
    /// Center of projection
    vec3<float> m_p;

    /// Coordinate basis
    CameraBasis m_basis;

    /// Height of the focal plan
    float m_focal_h;

    /// Distance to the focal plane
    float m_focal_d;

    /// Aperture diameter
    float m_a;

    /// Camera model
    CameraModel m_model;

    /// Counter for aperture sample placement
    static const unsigned int rng_val_aperture = 0x983abc12;

    /// Width of the output image (in pixels)
    unsigned int m_width;

    /// Height of the output image (in pixels)
    unsigned int m_height;

    /// Random number seed
    unsigned int m_seed;
};

} // namespace fresnel

#undef DEVICE

#endif
