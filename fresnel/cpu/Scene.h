// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef SCENE_H_
#define SCENE_H_

#include "embree_platform.h"
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <pybind11/pybind11.h>

#include "Device.h"

namespace fresnel { namespace cpu {

// Thin wrapper for RTCScene
/* Handle construction and deletion of the scene, and python lifetime as an exported class.
*/
class Scene
    {
    public:
        //! Constructor
        Scene(std::shared_ptr<Device> device);
        //! Destructor
        ~Scene();

        //! Access the RTCScene
        RTCScene& getScene()
            {
            return m_scene;
            }

    private:
        RTCScene m_scene;                   //!< Store the scene
        std::shared_ptr<Device> m_device;   //!< The device the scene is attached to
    };

//! Export Scene to python
void export_Scene(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
