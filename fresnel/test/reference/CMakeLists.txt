set(files test_camera_perspective.test_focal_distance.png
          test_camera_perspective.test_focal_length.png
          test_camera_perspective.test_f_stop.png
          test_camera_perspective.test_height.png
          test_camera_perspective.test_pathtrace.png
          test_camera_perspective.test_render.png
          test_geometry_box.test_box_color.png
          test_geometry_box.test_box_radius.png
          test_geometry_box.test_color.png
          test_geometry_box.test_radius.png
          test_geometry_box.test_render0.png
          test_geometry_box.test_render1.png
          test_geometry_box.test_render2.png
          test_geometry_box.test_render3.png
          test_geometry_box.test_render4.png
          test_geometry_box.test_render5.png
          test_geometry_box.test_render.png
          test_geometry_clyinder.test_color.png
          test_geometry_clyinder.test_outline.png
          test_geometry_clyinder.test_position.png
          test_geometry_clyinder.test_radius.png
          test_geometry_clyinder.test_render.png
          test_geometry_convex_polyhedron.test_face_color.png
          test_geometry_convex_polyhedron.test_outline.png
          test_geometry_convex_polyhedron.test_render.png
          test_geometry_mesh.test_color_interp.png
          test_geometry_mesh.test_multiple.png
          test_geometry_mesh.test_outline.png
          test_geometry_mesh.test_render.png
          test_geometry_polygon.test_angle.png
          test_geometry_polygon.test_color.png
          test_geometry_polygon.test_outline.png
          test_geometry_polygon.test_position.png
          test_geometry_polygon.test_render.png
          test_geometry_polygon.test_rounded.png
          test_geometry_sphere.test_color.png
          test_geometry_sphere.test_outline.png
          test_geometry_sphere.test_position.png
          test_geometry_sphere.test_radius.png
          test_geometry_sphere.test_render.png
          test_material.test_color.png
          test_material.test_metal.png
          test_material.test_primitive_color_mix.png
          test_material.test_roughness.png
          test_material.test_set_material.png
          test_material.test_solid.png
          test_material.test_specular.png
          test_outline_material.test_color.png
          test_outline_material.test_primitive_color_mix.png
          test_outline_material.test_set_material.png
          test_outline_material.test_solid.png
          test_scene.test_camera.png
          test_scene.test_light_dir.png
          test_scene.test_multiple_geometries1.png
          test_scene.test_multiple_geometries2.png
          test_scene.test_multiple_geometries3.png
          test_scene.test_multiple_geometries4.png
          test_tracer_direct.test_render_aa.png
          test_tracer_direct.test_render.png
          test_tracer_path.test_render.png)

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/test/reference
       )

copy_files_to_build("${files}" "fresnel_test_reference" "*.png")
