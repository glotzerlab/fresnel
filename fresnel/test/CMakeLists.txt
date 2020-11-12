set(files conftest.py
          pytest.ini
          test_camera_orthographic.py
          test_camera_perspective.py
          test_device.py
          test_geometry_box.py
          test_geometry_convex_polyhedron.py
          test_geometry_cylinder.py
          test_geometry_mesh.py
          test_geometry_polygon.py
          test_geometry_sphere.py
          test_light.py
          test_material.py
          test_outline_material.py
          test_scene.py
          test_tracer_direct.py
          test_tracer_path.py)

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/test
       )

copy_files_to_build("${files}" "fresnel_test" "*.py")

add_subdirectory(reference)
