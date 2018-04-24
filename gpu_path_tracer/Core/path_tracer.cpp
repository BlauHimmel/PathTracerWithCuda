#include "Core\path_tracer.h"

path_tracer::~path_tracer()
{
	if (m_is_initiated)
	{
		release_image(m_image);
	}
	path_tracer_kernel_memory_free(
		m_not_absorbed_colors_device,
		m_accumulated_colors_device,
		m_rays_device,
		m_energy_exist_pixels_device,
		m_scatterings_device
	);
}

std::vector<std::string> path_tracer::init(render_camera* render_camera, config_parser* config, const std::string& scene_file_directory)
{
	m_config = config;
	m_render_camera = render_camera;
	m_image = create_image(static_cast<int>(render_camera->resolution.x), static_cast<int>(render_camera->resolution.y));
	path_tracer_kernel_memory_allocate(
		&m_not_absorbed_colors_device,
		&m_accumulated_colors_device,
		&m_rays_device,
		&m_energy_exist_pixels_device,
		&m_scatterings_device,
		m_image->pixel_count
	);

	configuration* config_device = m_config->get_config_device_ptr();
	bvh_build_config::bvh_leaf_node_triangle_num = config_device->bvh_leaf_node_triangle_num;
	bvh_build_config::bvh_bucket_max_divide_internal_num = config_device->bvh_bucket_max_divide_internal_num;
	bvh_build_config::bvh_build_block_size = config_device->bvh_build_block_size;

	return m_scene.set_scene_file_directory(scene_file_directory);
}

image* path_tracer::render()
{
	if (m_is_initiated)
	{
		m_image->pass_counter++;

		path_tracer_kernel(
			m_scene.get_mesh_num(),
			m_scene.get_bvh_node_device_ptr(),
			m_scene.get_triangles_device_ptr(),
			m_scene.get_sphere_num(),
			m_scene.get_sphere_device_ptr(),
			m_image->pixel_count,
			m_image->pixels_device,
			m_image->pixels_256_device,
			m_image->pass_counter,
			m_render_camera,
			m_scene.get_cube_map_device_ptr(),
			m_not_absorbed_colors_device,
			m_accumulated_colors_device,
			m_rays_device,
			m_energy_exist_pixels_device,
			m_scatterings_device,
			m_scene.get_mesh_texture_device_ptr(),
			m_config->get_config_device_ptr()
		);

		return m_image;
	}
	else
	{
		return nullptr;
	}
}

void path_tracer::clear()
{
	if (m_is_initiated)
	{
		reset_image(m_image);
	}
}

void path_tracer::render_ui()
{
	if (!m_is_initiated)
	{
		return;
	}

	char buffer[2048];

	bool is_sphere_modified = false;

	if (m_scene.get_sphere_num() > 0 && ImGui::TreeNode("Sphere"))
	{
		for (auto i = 0; i < m_scene.get_sphere_num(); i++)
		{
			sprintf(buffer, "Sphere-%d", i + 1);
			if (ImGui::TreeNode(buffer))
			{
				bool is_modified = false;

				ImGui::Separator();

				sphere sphere = m_scene.get_sphere(i);

				float position[3] = { sphere.center.x, sphere.center.y ,sphere.center.z };
				float radius = sphere.radius;

				ImGui::Text("Base:");
				is_modified = is_modified || ImGui::DragFloat3("Position", position, 0.001f);
				is_modified = is_modified || ImGui::DragFloat("Radius", &radius, 0.001f, 0.001f, INFINITY);

				ImGui::Separator();

				material mat = sphere.mat;

				float diffuse[3] = { mat.diffuse_color.x, mat.diffuse_color.y, mat.diffuse_color.z };
				float specular[3] = { mat.specular_color.x, mat.specular_color.y, mat.specular_color.z };
				float emission[3] = { mat.emission_color.x, mat.emission_color.y, mat.emission_color.z };
				bool is_transparent = mat.is_transparent;
				float roughness = mat.roughness;

				ImGui::Text("Material:");
				is_modified = is_modified || ImGui::ColorEdit3("Diffuse", diffuse);
				is_modified = is_modified || ImGui::ColorEdit3("Specular", specular);
				is_modified = is_modified || ImGui::ColorEdit3("Emission", emission);
				is_modified = is_modified || ImGui::Checkbox("Transparent", &is_transparent);
				is_modified = is_modified || ImGui::SliderFloat("Roughness", &roughness, 0.0f, 1.0f);

				float refraction_index = mat.medium.refraction_index;
				float extinction_coefficient = mat.medium.extinction_coefficient;

				is_modified = is_modified || ImGui::DragFloat("Refraction Index", &refraction_index, 0.001f);
				if (!is_transparent)
				{
					is_modified = is_modified || ImGui::DragFloat("Extinction Coefficient", &extinction_coefficient, 0.001f);
				}

				float absorption_coefficient[] = { mat.medium.scattering.absorption_coefficient.x, mat.medium.scattering.absorption_coefficient.y, mat.medium.scattering.absorption_coefficient.z };
				float reduced_scattering_coefficient = mat.medium.scattering.reduced_scattering_coefficient.x;

				is_modified = is_modified || ImGui::DragFloat3("Absorption Coefficient", absorption_coefficient, 0.001f);
				is_modified = is_modified || ImGui::DragFloat("Reduced Scattering Coefficient", &reduced_scattering_coefficient, 0.001f);

				ImGui::Separator();

				if (is_modified)
				{
					is_sphere_modified = true;

					sphere.center = make_float3(position[0], position[1], position[2]);
					sphere.radius = radius;

					mat.diffuse_color = make_float3(diffuse[0], diffuse[1], diffuse[2]);
					mat.specular_color = make_float3(specular[0], specular[1], specular[2]);
					mat.emission_color = make_float3(emission[0], emission[1], emission[2]);
					mat.is_transparent = is_transparent;
					mat.roughness = roughness;

					mat.medium.refraction_index = refraction_index;
					if (!is_transparent)
					{
						mat.medium.extinction_coefficient = extinction_coefficient;
					}
					else
					{
						mat.medium.extinction_coefficient = 0.0f;
					}
					mat.medium.scattering.absorption_coefficient = make_float3(absorption_coefficient[0], absorption_coefficient[1], absorption_coefficient[2]);
					mat.medium.scattering.reduced_scattering_coefficient = make_float3(reduced_scattering_coefficient, reduced_scattering_coefficient, reduced_scattering_coefficient);

					sphere.mat = mat;

					m_scene.set_sphere_device(i, sphere);
				}

				ImGui::TreePop();
			}
		}

		ImGui::TreePop();
	}

	bool is_triangle_mesh_modified = false;

	if (m_scene.get_mesh_num() > 0 && ImGui::TreeNode("Mesh"))
	{
		for (auto i = 0; i < m_scene.get_mesh_num(); i++)
		{
			sprintf(buffer, "Mesh-%d", i + 1);
			if (ImGui::TreeNode(buffer))
			{
				bool is_bvh_update = false;
				bool is_rotate = false;
				bool is_material_modify = false;
				bool is_rotate_apply = false;
				bool is_rotate_clear = false;

				ImGui::Separator();

				float3 position = m_scene.get_mesh_position(i);
				float3 scale = m_scene.get_mesh_scale(i);
				float3 rotate = m_scene.get_mesh_rotate(i);;

				ImGui::Text("Base:");
				sprintf(buffer, "Vertices: %d", m_scene.get_mesh_vertices_num(i));
				ImGui::Text(buffer);
				sprintf(buffer, "Facets: %d", m_scene.get_mesh_triangle_num(i));
				ImGui::Text(buffer);
				sprintf(buffer, "Shapes: %d", m_scene.get_mesh_shape_num(i));
				ImGui::Text(buffer);

				is_bvh_update = is_bvh_update || ImGui::DragFloat3("Position", &position.x, 0.001f);
				is_bvh_update = is_bvh_update || ImGui::DragFloat3("Scale", &scale.x, 0.000001f, 0.000001f, INFINITY, "%.6f");
				is_rotate = is_rotate || ImGui::DragFloat3("Rotate", &rotate.x);

				ImGui::SameLine();
				is_rotate_apply = ImGui::Button("Apply");

				ImGui::SameLine();
				is_rotate_clear = ImGui::Button("Clear");
				is_rotate = is_rotate || is_rotate_clear;
				if (is_rotate_clear)
				{
					rotate = m_scene.get_mesh_rotate_applied(i);
				}

				ImGui::Separator();

				std::vector<material> mats = m_scene.get_mesh_material(i);

				for (auto j = 0; j < mats.size(); j++)
				{
					sprintf(buffer, "Mateiral-%d", j + 1);

					if (ImGui::TreeNode(buffer))
					{
						bool is_this_material_modified = false;

						float diffuse[3] = { mats[j].diffuse_color.x, mats[j].diffuse_color.y, mats[j].diffuse_color.z };
						float specular[3] = { mats[j].specular_color.x, mats[j].specular_color.y, mats[j].specular_color.z };
						float emission[3] = { mats[j].emission_color.x, mats[j].emission_color.y, mats[j].emission_color.z };
						bool is_transparent = mats[j].is_transparent;
						float roughness = mats[j].roughness;

						is_this_material_modified = is_this_material_modified || ImGui::ColorEdit3("Diffuse", diffuse);
						is_this_material_modified = is_this_material_modified || ImGui::ColorEdit3("Specular", specular);
						is_this_material_modified = is_this_material_modified || ImGui::ColorEdit3("Emission", emission);
						is_this_material_modified = is_this_material_modified || ImGui::Checkbox("Transparent", &is_transparent);
						is_this_material_modified = is_this_material_modified || ImGui::SliderFloat("Roughness", &roughness, 0.0f, 1.0f);

						float refraction_index = mats[j].medium.refraction_index;
						float extinction_coefficient = mats[j].medium.extinction_coefficient;

						is_this_material_modified = is_this_material_modified || ImGui::DragFloat("Refraction Index", &refraction_index, 0.001f);
						if (!is_transparent)
						{
							is_this_material_modified = is_this_material_modified || ImGui::DragFloat("Extinction Coefficient", &extinction_coefficient, 0.001f);
						}

						float absorption_coefficient[] = { mats[j].medium.scattering.absorption_coefficient.x, mats[j].medium.scattering.absorption_coefficient.y, mats[j].medium.scattering.absorption_coefficient.z };
						float reduced_scattering_coefficient = mats[j].medium.scattering.reduced_scattering_coefficient.x;

						is_this_material_modified = is_this_material_modified || ImGui::DragFloat3("Absorption Coefficient", absorption_coefficient, 0.001f);
						is_this_material_modified = is_this_material_modified || ImGui::DragFloat("Reduced Scattering Coefficient", &reduced_scattering_coefficient, 0.001f);

						is_material_modify = is_material_modify || is_this_material_modified;

						if (is_this_material_modified)
						{
							material new_mat;

							new_mat.diffuse_color = make_float3(diffuse[0], diffuse[1], diffuse[2]);
							new_mat.specular_color = make_float3(specular[0], specular[1], specular[2]);
							new_mat.emission_color = make_float3(emission[0], emission[1], emission[2]);
							new_mat.is_transparent = is_transparent;
							new_mat.roughness = roughness;
							new_mat.medium.refraction_index = refraction_index;
							if (!is_transparent)
							{
								new_mat.medium.extinction_coefficient = extinction_coefficient;
							}
							else
							{
								new_mat.medium.extinction_coefficient = 0.0f;
							}
							new_mat.medium.scattering.absorption_coefficient = make_float3(absorption_coefficient[0], absorption_coefficient[1], absorption_coefficient[2]);
							new_mat.medium.scattering.reduced_scattering_coefficient = make_float3(reduced_scattering_coefficient, reduced_scattering_coefficient, reduced_scattering_coefficient);
							new_mat.diffuse_texture_id = mats[j].diffuse_texture_id;
							new_mat.specular_texture_id = mats[j].specular_texture_id;

							mats[j] = new_mat;
						}

						ImGui::Separator();
						ImGui::TreePop();
					}
				}

				if (is_material_modify)
				{
					is_triangle_mesh_modified = true;
					m_scene.set_mesh_material_device(i, mats);
				}

				if (is_rotate)
				{
					is_triangle_mesh_modified = true;
					m_scene.set_mesh_rotate(i, rotate);
				}

				if (is_bvh_update)
				{
					is_triangle_mesh_modified = true;
					std::function<void(const glm::mat4&, const glm::mat4&, bvh_node_device*, bvh_node_device*)> bvh_update_function =
						BVH_BUILD_METHOD update_bvh;
					m_scene.set_mesh_transform_device(
						i,
						position,
						clamp(scale, make_float3(0.000001f, 0.000001f, 0.000001f), make_float3(INFINITY, INFINITY, INFINITY)),
						bvh_update_function
					);
				}

				if (is_rotate_apply)
				{
					is_triangle_mesh_modified = true;
					m_scene.apply_mesh_rotate(i);
				}

				ImGui::TreePop();
			}
		}

		ImGui::TreePop();
	}

	if (is_sphere_modified || is_triangle_mesh_modified)
	{
		clear();
	}
}

bool path_tracer::init_scene_device_data(int index)
{
	double time;
	printf("[Info]Load scene data...\n");
	TIME_COUNT_CALL_START();
	if (!m_scene.load_scene(index))
	{
		m_is_initiated = false;
		std::cout << "[Error]Load scene failed!" << std::endl;
		return false;
	}
	if (!m_scene.create_scene_data_device())
	{
		m_is_initiated = false;
		std::cout << "[Error]Copy scene data on GPU failed!" << std::endl;
		m_scene.release_scene_data_device();
		return false;
	}
	TIME_COUNT_CALL_END(time);
	printf("[Info]Load scene completed, total time consuming: %.4f ms\n", time);
	m_is_initiated = true;
	return true;
}

void path_tracer::release_scene_device_data()
{
	if (!m_is_initiated)
	{
		return;
	}

	m_scene.release_scene_data_device();
	m_scene.unload_scene();
}
