#pragma once

#ifndef __TRIANGLE_MESH__
#define __TRIANGLE_MESH__

#include <cuda_runtime.h>
#include <iostream>
#include <functional>
#include <time.h>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include "lib\tiny_obj_loader\tiny_obj_loader.h"
#include "Core\triangle.h"
#include "Core\material.h"
#include "Others\utilities.hpp"
#include "Math\cuda_math.hpp"
#include "Bvh\bvh.h"

class triangle_mesh
{
private:
	//============================================
	std::vector<triangle> m_triangles;

	std::vector<int> m_mesh_triangles_num;
	std::vector<int> m_mesh_vertices_num;

	std::vector<float3> m_mesh_position;
	std::vector<float3> m_mesh_scale;
	std::vector<float3> m_mesh_rotate;
	std::vector<float3> m_mesh_rotate_applied;

	std::vector<glm::mat4> m_mesh_initial_transform;
	std::vector<glm::mat4> m_mesh_current_transform;

	std::vector<material*> m_mesh_material;
	std::vector<std::string> m_mesh_name;

	int m_mesh_num = 0;
	//============================================

	bool m_is_loaded = false;

	triangle* m_mesh_device = nullptr;
	material* m_mat_device = nullptr;

	bvh_node_device** m_mesh_bvh_initial_device = nullptr;
	bvh_node_device** m_mesh_bvh_transformed_device = nullptr;

public:
	bool load_obj(const std::string& filename, const float3& position, const float3& scale, const float3& rotate, material* mat);
	void unload_obj();

	void set_material_device(int index, const material& mat);
	void set_transform_device(
		int index, 
		const float3& position, 
		const float3& scale, 
		std::function<void(const glm::mat4&, glm::mat4&, bvh_node_device*, bvh_node_device*)> bvh_update_function
	);
	void set_rotate(int index, const float3& rotate);
	void apply_rotate(int index);

	int get_total_triangle_num() const;
	int get_mesh_num() const;
	float3 get_position(int index) const;
	float3 get_scale(int index) const;
	float3 get_rotate(int index) const;
	float3 get_rotate_applied(int index) const;
	material get_material(int index) const;
	int get_triangle_num(int index) const;
	int get_vertex_num(int index) const;
	triangle* get_triangles_device() const;
	bvh_node_device** get_bvh_node_device() const;

	bool create_bvh_device_data();
	void release_bvh_device_data();

	bool create_mesh_device_data();
	void release_mesh_device_data();
};

#endif // !__TRIANGLE_MESH__