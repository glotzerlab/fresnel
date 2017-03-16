// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef SCENE_H_
#define SCENE_H_

#include <optixu/optixpp_namespace.h>
#include <pybind11/pybind11.h>

#include "Device.h"
#include "common/Camera.h"

namespace fresnel { namespace gpu {

//! Manage the root scene graph object
/*! OptiX does not define a concrete notion of a scene. We create one in Fresnel by managing the root object of the
    scene graph as a Scene, for compatibility with the cpu::Scene API. The root object is a Geometry Group which can
    hold any number of geometry instances.

    The Scene also manages an acceleration structure for all of the primitives. Whenever a child object is modified,
    added, or removed, they must mark the acceleration structure dirty.

    A given Scene also has an associated camera, background color, and background alpha. The camera is used by the
    Tracer to generate rays into the Scene. The background color and alpha are the resulting color output by the
    Tracer when a ray fails to hit geometry in the Scene.

    Scene will eventually support multiple lights. As a temporary API, Scene stores a single light direction.

    In OptiX, the root object of the scene does not accept parameters. Camera parameters, background colors, etc...
    need to be set by the Tracer at the context level prior to launching the ray generation program.
*/
class Scene
    {
    public:
        //! Constructor
        Scene(std::shared_ptr<Device> device);
        //! Destructor
        ~Scene();

        //! Access the root object
        optix::GeometryGroup& getRoot()
            {
            return m_root;
            }

        //! Access the acceleration structure
        optix::Acceleration& getAccel()
            {
            return m_accel;
            }

        //! Access the Device
        std::shared_ptr<Device> getDevice()
            {
            return m_device;
            }

        //! Add a geometry instance to the scene
        void addGeometry(optix::GeometryInstance inst)
            {
            m_root->addChild(inst);
            m_accel->markDirty();
            }

        //! Remove a geometry instance from the scene
        void removeGeometry(optix::GeometryInstance inst)
            {
            m_root->removeChild(inst);
            m_accel->markDirty();
            }

        //! Update acceleration structures
        /*! Call when any geometry in this scene is modified
        */
        void update()
            {
            m_accel->markDirty();
            }

        //! Set the camera
        void setCamera(const UserCamera& camera)
            {
            m_camera = camera;
            }

        //! Get the camera
        UserCamera& getCamera()
            {
            return m_camera;
            }

        //! Set the background color
        void setBackgroundColor(const RGB<float>& c)
            {
            m_background_color = c;
            }

        //! Get the background color
        RGB<float> getBackgroundColor() const
            {
            return m_background_color;
            }

        //! Set the background alpha
        void setBackgroundAlpha(float a)
            {
            m_background_alpha = a;
            }

        //! Get the background alpha
        float getBackgroundAlpha() const
            {
            return m_background_alpha;
            }

        //! Set the light direction
        void setLightDirection(const vec3<float>& v)
            {
            vec3<float> l = v;
            l = l / sqrtf(dot(l,l));
            m_light_direction = l;
            }

        //! Get the light direction
        vec3<float> getLightDirection() const
            {
            return m_light_direction;
            }

    private:
        optix::GeometryGroup m_root;        //!< Store the scene root object
        optix::Acceleration m_accel;        //!< Store the acceleration structure
        std::shared_ptr<Device> m_device;   //!< The device the scene is attached to

        RGB<float> m_background_color;              //!< The background color
        float m_background_alpha;                   //!< Background alpha
        vec3<float> m_light_direction;              //!< The light direction
        UserCamera m_camera;                        //!< The camera
    };

//! Export Scene to python
void export_Scene(pybind11::module& m);

} } // end namespace fresnel::gpu

#endif
