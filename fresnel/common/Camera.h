// Copyright (c) 2016-2022 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

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

namespace fresnel
    {
/** Camera models.

    Used to select the camera model in Camera.
*/
enum class CameraModel
    {
    orthographic,
    perspective
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

    Perspective models also use:

    * f: The focal length of modelled lens.
    * f_stop: Set the aperture diameter to `f/f_stop`.

    All convenience methods and derived properties are implemented in Python.
*/
struct UserCamera
    {
    DEVICE UserCamera()
        {
        position = vec3<float>(0, 0, 0);
        look_at = vec3<float>(0, 0, 1);
        up = vec3<float>(0, 1, 0);
        h = 1;
        f = 1;
        f_stop = 1;
        focus_distance = 1;
        model = CameraModel::orthographic;
        }

    vec3<float> position;
    vec3<float> look_at;
    vec3<float> up;
    float h;
    float f;
    float f_stop;
    float focus_distance;
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
    DEVICE CameraBasis() { }
    DEVICE explicit CameraBasis(const UserCamera& user) : v(user.up)
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

/** Implementation of the camera models.

    RayGen generates image sample locations in fractional coordinates that range from -0.5 to 0.5 in
    the y direction and from -0.5*aspect to 0.5*aspect in the x direction, where aspect is the
    aspect ratio of the image. Given a sample location on the image plane, Camera generates a ray
    (origin and normalized direction) that should be sampled.

    Construct a Camera from a UserCamera. UserCamera provides all of the properties needed to define
    the camera model. Camera preprocesses some of these inputs into values usable when generating
    rays.

    Some camera models (such as perspective) utilize random sampling. The generate() method takes a
    pixel location, a user random number seed value, and a sample index.
*/
class Camera
    {
    public:
    DEVICE Camera() { }

    /** Construct from a UserCamera.

        @param user Camera parameters.
        @param width Width of the image in pixels.
        @param height Height of the image in pixels.
        @param seed Random number seed.
    */
    DEVICE explicit Camera(const UserCamera& user,
                           unsigned int width,
                           unsigned int height,
                           unsigned int seed,
                           bool sample_aa = true)
        : m_p(user.position), m_basis(user), m_focal_d(user.focus_distance),
          m_a(user.f / user.f_stop), m_model(user.model), m_width(width), m_height(height),
          m_seed(seed), m_sample_aa(sample_aa)
        {
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
        @param i Pixel index in the x direction.
        @param j Pixel index in the y direction.
        @param sample Index of the sample to generate.
    */
    DEVICE void generateRay(vec3<float>& origin,
                            vec3<float>& direction,
                            unsigned int i,
                            unsigned int j,
                            unsigned int sample) const
        {
        vec2<float> s = importanceSampleAA(i, j, sample);

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

        if (m_model == CameraModel::perspective)
            {
            // create the philox unique key for this RNG which includes the pixel ID and the random
            // seed
            unsigned int pixel = j * m_width + i;
            r123::Philox4x32::ukey_type rng_key = {{pixel, m_seed}};

            // generate a random point in a circle
            r123::Philox4x32 rng;
            r123::Philox4x32::ctr_type rng_counter = {{0, 0, sample, rng_val_aperture}};
            r123::Philox4x32::ctr_type rng_u = rng(rng_counter, rng_key);
            float theta = r123::u01<float>(rng_u.v[0]) * 2.0f * float(M_PI);
            float r = r123::u01<float>(rng_u.v[1]) * m_a * 0.5f;

            vec3<float> offset = r * cosf(theta) * m_basis.u + r * sinf(theta) * m_basis.v;
            origin = m_p + offset;
            }

        direction = C - origin;
        direction *= 1.0f / sqrtf(dot(direction, direction));
        }

    /// Get the camera basis.
    DEVICE const CameraBasis& getBasis() const
        {
        return m_basis;
        }

    private:
    /** Importance sample pixel locations for anti-aliasing

        @param i Pixel index in the x direction.
        @param j Pixel index in the y direction.
        @param sample Index of the sample to generate.

        Given the sample index, importance sample the tent filter to produce anti-aliased output.
    */
    DEVICE vec2<float> importanceSampleAA(unsigned int i, unsigned int j, unsigned int sample) const
        {
        float i_f, j_f;

        if (m_sample_aa)
            {
            // create the philox unique key for this RNG which includes the pixel ID and the random
            // seed
            unsigned int pixel = j * m_width + i;
            r123::Philox4x32::ukey_type rng_uk = {{pixel, m_seed}};

            // generate 2 random numbers from 0 to 2
            r123::Philox4x32 rng;
            r123::Philox4x32::ctr_type rng_counter = {{0, 0, sample, rng_val_aa}};
            r123::Philox4x32::ctr_type rng_u = rng(rng_counter, rng_uk);
            float r1 = r123::u01<float>(rng_u.v[0]) * 2.0f;
            float r2 = r123::u01<float>(rng_u.v[1]) * 2.0f;

            // use important sampling to sample the tent filter
            float dx, dy;
            if (r1 < 1.0f)
                dx = sqrtf(r1) - 1.0f;
            else
                dx = 1.0f - sqrtf(2.0f - r1);

            if (r2 < 1.0f)
                dy = sqrtf(r2) - 1.0f;
            else
                dy = 1.0f - sqrtf(2.0f - r2);

            i_f = float(i) + 0.5f + dx * m_aa_w;
            j_f = float(j) + 0.5f + dy * m_aa_w;
            }
        else
            {
            i_f = float(i) + 0.5f;
            j_f = float(j) + 0.5f;
            }

        // determine the viewing plane relative coordinates
        float ys = -1.0f * (j_f / float(m_height) - 0.5f);
        float xs = i_f / float(m_height) - 0.5f * float(m_width) / float(m_height);
        return vec2<float>(xs, ys);
        }

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

    /// Width of the output image (in pixels)
    unsigned int m_width;

    /// Height of the output image (in pixels)
    unsigned int m_height;

    /// Random number seed
    unsigned int m_seed;

    /// Flag to enable antialiasing samples
    bool m_sample_aa;

    /// Counter for aperture sample placement
    static constexpr unsigned int rng_val_aperture = 0x983abc12;

    /// Counter for anti-aliasing samples
    static constexpr unsigned int rng_val_aa = 0x22ab5871;

    /// Width of the anti-aliasing filter (in pixels)
    static constexpr float m_aa_w = 0.707106781f;
    };

    } // namespace fresnel

#undef DEVICE

#endif
