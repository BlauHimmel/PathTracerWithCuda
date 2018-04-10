#pragma once

#ifndef __TRIANGLE_MESH__
#define __TRIANGLE_MESH__

#include <cuda_runtime.h>
#include <iostream>
#include <functional>
#include <time.h>
#include <glm\glm.hpp>
#include "lib\tiny_obj_loader\tiny_obj_loader.h"
#include "triangle.hpp"
#include "material.hpp"
#include "utilities.hpp"
#include "cuda_math.hpp"
#include "bvh.h"

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

inline bool triangle_mesh::load_obj(const std::string& filename, const float3& position, const float3& scale, const float3& rotate, material* mat)
{
	if (mat == nullptr)
	{
		return false;
	}

	std::string mesh_name =filename.substr(filename.find_last_of('\\') + 1);

	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	tinyobj::attrib_t attrib; 
	std::string error;

	int triangle_num = 0;

	std::cout << "[Info]Loading file " << mesh_name << "...." << std::endl;

	bool is_success = tinyobj::LoadObj(&attrib, &shapes, &materials, &error, filename.c_str());

	if (!error.empty())
	{
		std::cout << "[TinyObj]\n" << error.substr(0, error.find_last_of('\n')) << std::endl;
	}

	if (!is_success)
	{
		std::cout << "[Info]Load file " << mesh_name << " failed." << std::endl;
		return false;
	}

	for (auto i = 0; i < shapes.size(); i++)
	{
		for (auto num : shapes[i].mesh.num_face_vertices)
		{
			if (num != 3)
			{
				std::cout << "[Error]" << mesh_name << " is not a triangle mesh." << std::endl;
				return false;
			}
		}

		for (auto f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++)
		{
			float3 vertex[3];
			float3 normal[3];
			tinyobj::index_t index[3];

			index[0] = shapes[i].mesh.indices[f * 3];
			index[1] = shapes[i].mesh.indices[f * 3 + 1];
			index[2] = shapes[i].mesh.indices[f * 3 + 2];

			vertex[0] = make_float3(
				attrib.vertices[index[0].vertex_index * 3],
				attrib.vertices[index[0].vertex_index * 3 + 1],
				attrib.vertices[index[0].vertex_index * 3 + 2]
			);

			vertex[1] = make_float3(
				attrib.vertices[index[1].vertex_index * 3],
				attrib.vertices[index[1].vertex_index * 3 + 1],
				attrib.vertices[index[1].vertex_index * 3 + 2]
			);
			
			vertex[2] = make_float3(
				attrib.vertices[index[2].vertex_index * 3],
				attrib.vertices[index[2].vertex_index * 3 + 1],
				attrib.vertices[index[2].vertex_index * 3 + 2]
			);

			normal[0] = make_float3(
				attrib.normals[index[0].normal_index * 3],
				attrib.normals[index[0].normal_index * 3 + 1],
				attrib.normals[index[0].normal_index * 3 + 2]
			);

			normal[1] = make_float3(
				attrib.normals[index[1].normal_index * 3],
				attrib.normals[index[1].normal_index * 3 + 1],
				attrib.normals[index[1].normal_index * 3 + 2]
			);

			normal[2] = make_float3(
				attrib.normals[index[2].normal_index * 3],
				attrib.normals[index[2].normal_index * 3 + 1],
				attrib.normals[index[2].normal_index * 3 + 2]
			);

			glm::vec4 vertex0_vec4 = glm::vec4(vertex[0].x, vertex[0].y, vertex[0].z, 1.0f);
			glm::vec4 vertex1_vec4 = glm::vec4(vertex[1].x, vertex[1].y, vertex[1].z, 1.0f);
			glm::vec4 vertex2_vec4 = glm::vec4(vertex[2].x, vertex[2].y, vertex[2].z, 1.0f);

			glm::vec4 normal0_vec4 = glm::vec4(normal[0].x, normal[0].y, normal[0].z, 0.0f);
			glm::vec4 normal1_vec4 = glm::vec4(normal[1].x, normal[1].y, normal[1].z, 0.0f);
			glm::vec4 normal2_vec4 = glm::vec4(normal[2].x, normal[2].y, normal[2].z, 0.0f);

			glm::mat4 rotate_mat(1.0f);
			rotate_mat = glm::rotate(rotate_mat, glm::radians(rotate.z), glm::vec3(0.0f, 0.0f, 1.0f));
			rotate_mat = glm::rotate(rotate_mat, glm::radians(rotate.y), glm::vec3(0.0f, 1.0f, 0.0f));
			rotate_mat = glm::rotate(rotate_mat, glm::radians(rotate.x), glm::vec3(1.0f, 0.0f, 0.0f));
			glm::mat4 inverse_transpose_rotate_mat = glm::transpose(glm::inverse(rotate_mat));

			vertex0_vec4 = rotate_mat * vertex0_vec4;
			vertex1_vec4 = rotate_mat * vertex1_vec4;
			vertex2_vec4 = rotate_mat * vertex2_vec4;

			normal0_vec4 = inverse_transpose_rotate_mat * normal0_vec4;
			normal1_vec4 = inverse_transpose_rotate_mat * normal1_vec4;
			normal2_vec4 = inverse_transpose_rotate_mat * normal2_vec4;

			vertex[0] = make_float3(vertex0_vec4.x, vertex0_vec4.y, vertex0_vec4.z);
			vertex[1] = make_float3(vertex1_vec4.x, vertex1_vec4.y, vertex1_vec4.z);
			vertex[2] = make_float3(vertex2_vec4.x, vertex2_vec4.y, vertex2_vec4.z);
			normal[0] = normalize(make_float3(normal0_vec4.x, normal0_vec4.y, normal0_vec4.z));
			normal[1] = normalize(make_float3(normal1_vec4.x, normal1_vec4.y, normal1_vec4.z));
			normal[2] = normalize(make_float3(normal2_vec4.x, normal2_vec4.y, normal2_vec4.z));
			
			triangle triangle;
			triangle.vertex0 = vertex[0];
			triangle.vertex1 = vertex[1];
			triangle.vertex2 = vertex[2];
			triangle.normal0 = normal[0];
			triangle.normal1 = normal[1];
			triangle.normal2 = normal[2];
			triangle.mat = mat;

			m_triangles.push_back(triangle);

			triangle_num++;
		}
	}

	m_is_loaded = true;
	m_mesh_num++;

	int vertices_num = static_cast<int>(attrib.vertices.size() / 3);

	m_mesh_triangles_num.push_back(triangle_num);
	m_mesh_vertices_num.push_back(vertices_num);
	m_mesh_position.push_back(position);
	m_mesh_scale.push_back(scale);
	m_mesh_rotate.push_back(rotate);
	m_mesh_rotate_applied.push_back(rotate);

	glm::mat4 transform_mat(1.0f);
	transform_mat = glm::translate(transform_mat, glm::vec3(position.x, position.y, position.z));
	transform_mat = glm::scale(transform_mat, glm::vec3(scale.x, scale.y, scale.z));

	m_mesh_initial_transform.push_back(glm::inverse(transform_mat));
	m_mesh_current_transform.push_back(transform_mat);
	m_mesh_material.push_back(mat);
	m_mesh_name.push_back(mesh_name);

	std::cout << "[Info]Load file " << mesh_name << " succeeded. vertices : " << vertices_num << std::endl;

	return true;
}

inline void triangle_mesh::unload_obj()
{
	m_triangles.clear();
	m_triangles.shrink_to_fit();
	m_mesh_triangles_num.clear();
	m_mesh_triangles_num.shrink_to_fit();
	m_mesh_vertices_num.clear();
	m_mesh_vertices_num.shrink_to_fit();
	m_mesh_position.clear();
	m_mesh_position.shrink_to_fit();
	m_mesh_scale.clear();
	m_mesh_scale.shrink_to_fit();
	m_mesh_rotate.clear();
	m_mesh_rotate.shrink_to_fit();
	m_mesh_rotate_applied.clear();
	m_mesh_rotate_applied.shrink_to_fit();
	m_mesh_initial_transform.clear();
	m_mesh_initial_transform.shrink_to_fit();
	m_mesh_current_transform.clear();
	m_mesh_current_transform.shrink_to_fit();
	m_mesh_name.clear();
	m_mesh_name.shrink_to_fit();
	m_mesh_num = 0;

	for (auto mat : m_mesh_material)
	{
		SAFE_DELETE(mat);
	}
	m_mesh_material.clear();
	m_mesh_material.shrink_to_fit();

	m_is_loaded = false;
}

inline void triangle_mesh::set_material_device(int index, const material& mat)
{
	m_mat_device[index] = mat;
}

inline void triangle_mesh::set_transform_device(
	int index, 
	const float3& position,
	const float3& scale,
	std::function<void(const glm::mat4&, glm::mat4&, bvh_node_device*, bvh_node_device*)> bvh_update_function
)
{
	glm::mat4 transform_mat(1.0f);
	transform_mat = glm::translate(transform_mat, glm::vec3(position.x, position.y, position.z));
	transform_mat = glm::scale(transform_mat, glm::vec3(scale.x, scale.y, scale.z));

	m_mesh_current_transform[index] = transform_mat;
	m_mesh_position[index] = position;
	m_mesh_scale[index] = scale;

	int triangle_start_index = 0;
	for (auto i = 0; i < index; i++)
	{
		triangle_start_index += m_mesh_triangles_num[i];
	}

	for (auto i = 0; i < m_mesh_triangles_num[index]; i++)
	{
		float3 vertex0 = m_triangles[i + triangle_start_index].vertex0;
		float3 vertex1 = m_triangles[i + triangle_start_index].vertex1;
		float3 vertex2 = m_triangles[i + triangle_start_index].vertex2;

		float3 normal0 = m_triangles[i + triangle_start_index].normal0;
		float3 normal1 = m_triangles[i + triangle_start_index].normal1;
		float3 normal2 = m_triangles[i + triangle_start_index].normal2;

		glm::vec4 vertex0_vec4 = glm::vec4(vertex0.x, vertex0.y, vertex0.z, 1.0f);
		glm::vec4 vertex1_vec4 = glm::vec4(vertex1.x, vertex1.y, vertex1.z, 1.0f);
		glm::vec4 vertex2_vec4 = glm::vec4(vertex2.x, vertex2.y, vertex2.z, 1.0f);

		glm::vec4 normal0_vec4 = glm::vec4(normal0.x, normal0.y, normal0.z, 0.0f);
		glm::vec4 normal1_vec4 = glm::vec4(normal1.x, normal1.y, normal1.z, 0.0f);
		glm::vec4 normal2_vec4 = glm::vec4(normal2.x, normal2.y, normal2.z, 0.0f);

		vertex0_vec4 = m_mesh_current_transform[index] * vertex0_vec4;
		vertex1_vec4 = m_mesh_current_transform[index] * vertex1_vec4;
		vertex2_vec4 = m_mesh_current_transform[index] * vertex2_vec4;

		glm::mat4 inverse_transpose_transform_mat = glm::transpose(glm::inverse(m_mesh_current_transform[index]));
		normal0_vec4 = inverse_transpose_transform_mat * normal0_vec4;
		normal1_vec4 = inverse_transpose_transform_mat * normal1_vec4;
		normal2_vec4 = inverse_transpose_transform_mat * normal2_vec4;

		m_mesh_device[i + triangle_start_index].vertex0 = make_float3(vertex0_vec4.x, vertex0_vec4.y, vertex0_vec4.z);
		m_mesh_device[i + triangle_start_index].vertex1 = make_float3(vertex1_vec4.x, vertex1_vec4.y, vertex1_vec4.z);
		m_mesh_device[i + triangle_start_index].vertex2 = make_float3(vertex2_vec4.x, vertex2_vec4.y, vertex2_vec4.z);
		m_mesh_device[i + triangle_start_index].normal0 = normalize(make_float3(normal0_vec4.x, normal0_vec4.y, normal0_vec4.z));
		m_mesh_device[i + triangle_start_index].normal1 = normalize(make_float3(normal1_vec4.x, normal1_vec4.y, normal1_vec4.z));
		m_mesh_device[i + triangle_start_index].normal2 = normalize(make_float3(normal2_vec4.x, normal2_vec4.y, normal2_vec4.z));
	}
	
	bvh_update_function(m_mesh_initial_transform[index], m_mesh_current_transform[index], m_mesh_bvh_initial_device[index], m_mesh_bvh_transformed_device[index]);
}

inline void triangle_mesh::set_rotate(int index, const float3& rotate)
{
	m_mesh_rotate[index] = rotate;
}

inline void triangle_mesh::apply_rotate(int index)
{
	if (m_mesh_bvh_initial_device == nullptr)
	{
		return;
	}

	CUDA_CALL(cudaFree(m_mesh_bvh_initial_device[index]));

	int triangle_start_index = 0;
	for (auto j = 0; j < index; j++)
	{
		triangle_start_index += m_mesh_triangles_num[j];
	}

	for (auto i = 0; i < m_mesh_triangles_num[index]; i++)
	{
		float3 vertex0 = m_triangles[i + triangle_start_index].vertex0;
		float3 vertex1 = m_triangles[i + triangle_start_index].vertex1;
		float3 vertex2 = m_triangles[i + triangle_start_index].vertex2;

		float3 normal0 = m_triangles[i + triangle_start_index].normal0;
		float3 normal1 = m_triangles[i + triangle_start_index].normal1;
		float3 normal2 = m_triangles[i + triangle_start_index].normal2;

		glm::vec4 vertex0_vec4 = glm::vec4(vertex0.x, vertex0.y, vertex0.z, 1.0f);
		glm::vec4 vertex1_vec4 = glm::vec4(vertex1.x, vertex1.y, vertex1.z, 1.0f);
		glm::vec4 vertex2_vec4 = glm::vec4(vertex2.x, vertex2.y, vertex2.z, 1.0f);

		glm::vec4 normal0_vec4 = glm::vec4(normal0.x, normal0.y, normal0.z, 0.0f);
		glm::vec4 normal1_vec4 = glm::vec4(normal1.x, normal1.y, normal1.z, 0.0f);
		glm::vec4 normal2_vec4 = glm::vec4(normal2.x, normal2.y, normal2.z, 0.0f);

		glm::mat4 rotate_mat(1.0f); 
		rotate_mat = glm::rotate(rotate_mat, glm::radians(m_mesh_rotate[index].z - m_mesh_rotate_applied[index].z), glm::vec3(0.0f, 0.0f, 1.0f));
		rotate_mat = glm::rotate(rotate_mat, glm::radians(m_mesh_rotate[index].y - m_mesh_rotate_applied[index].y), glm::vec3(0.0f, 1.0f, 0.0f));
		rotate_mat = glm::rotate(rotate_mat, glm::radians(m_mesh_rotate[index].x - m_mesh_rotate_applied[index].x), glm::vec3(1.0f, 0.0f, 0.0f));
		glm::mat4 inverse_transpose_rotate_mat = glm::transpose(glm::inverse(rotate_mat));

		vertex0_vec4 = rotate_mat * vertex0_vec4;
		vertex1_vec4 = rotate_mat * vertex1_vec4;
		vertex2_vec4 = rotate_mat * vertex2_vec4;

		normal0_vec4 = inverse_transpose_rotate_mat * normal0_vec4;
		normal1_vec4 = inverse_transpose_rotate_mat * normal1_vec4;
		normal2_vec4 = inverse_transpose_rotate_mat * normal2_vec4;

		m_triangles[i + triangle_start_index].vertex0 = make_float3(vertex0_vec4.x, vertex0_vec4.y, vertex0_vec4.z);
		m_triangles[i + triangle_start_index].vertex1 = make_float3(vertex1_vec4.x, vertex1_vec4.y, vertex1_vec4.z);
		m_triangles[i + triangle_start_index].vertex2 = make_float3(vertex2_vec4.x, vertex2_vec4.y, vertex2_vec4.z);
		m_triangles[i + triangle_start_index].normal0 = normalize(make_float3(normal0_vec4.x, normal0_vec4.y, normal0_vec4.z));
		m_triangles[i + triangle_start_index].normal1 = normalize(make_float3(normal1_vec4.x, normal1_vec4.y, normal1_vec4.z));
		m_triangles[i + triangle_start_index].normal2 = normalize(make_float3(normal2_vec4.x, normal2_vec4.y, normal2_vec4.z));

		vertex0_vec4 = m_mesh_current_transform[index] * vertex0_vec4;
		vertex1_vec4 = m_mesh_current_transform[index] * vertex1_vec4;
		vertex2_vec4 = m_mesh_current_transform[index] * vertex2_vec4;

		glm::mat4 inverse_transpose_transform_mat = glm::transpose(glm::inverse(m_mesh_current_transform[index]));
		normal0_vec4 = inverse_transpose_transform_mat * normal0_vec4;
		normal1_vec4 = inverse_transpose_transform_mat * normal1_vec4;
		normal2_vec4 = inverse_transpose_transform_mat * normal2_vec4;

		m_mesh_device[i + triangle_start_index].vertex0 = make_float3(vertex0_vec4.x, vertex0_vec4.y, vertex0_vec4.z);
		m_mesh_device[i + triangle_start_index].vertex1 = make_float3(vertex1_vec4.x, vertex1_vec4.y, vertex1_vec4.z);
		m_mesh_device[i + triangle_start_index].vertex2 = make_float3(vertex2_vec4.x, vertex2_vec4.y, vertex2_vec4.z);
		m_mesh_device[i + triangle_start_index].normal0 = normalize(make_float3(normal0_vec4.x, normal0_vec4.y, normal0_vec4.z));
		m_mesh_device[i + triangle_start_index].normal1 = normalize(make_float3(normal1_vec4.x, normal1_vec4.y, normal1_vec4.z));
		m_mesh_device[i + triangle_start_index].normal2 = normalize(make_float3(normal2_vec4.x, normal2_vec4.y, normal2_vec4.z));
	}	

	m_mesh_initial_transform[index] = glm::inverse(m_mesh_current_transform[index]);

	double time;
	bvh_node* root;
	printf("[Info]Apply rotate, reconstructing bvh for mesh %s...\n", m_mesh_name[index].c_str());
	TIME_COUNT_CALL_START();
	root = BVH_BUILD_METHOD build_bvh(m_mesh_device + triangle_start_index, m_mesh_triangles_num[index], triangle_start_index);
	TIME_COUNT_CALL_END(time);
	printf("[Info]Completed, time consuming: %.4f ms\n", time);

	printf("[Info]Apply rotate, copy bvh data for mesh %s to GPU...\n", m_mesh_name[index].c_str());
	TIME_COUNT_CALL_START();
	m_mesh_bvh_initial_device[index] = BVH_BUILD_METHOD build_bvh_device_data(root);
	m_mesh_bvh_transformed_device[index] = m_mesh_bvh_initial_device[index] + m_mesh_bvh_initial_device[index]->next_node_index;
	TIME_COUNT_CALL_END(time);
	printf("[Info]Completed, time consuming: %.4f ms\n", time);

	BVH_BUILD_METHOD release_bvh(root);

	m_mesh_rotate_applied[index] = m_mesh_rotate[index];
}

inline int triangle_mesh::get_total_triangle_num() const
{
	return static_cast<int>(m_triangles.size());
}

inline int triangle_mesh::get_mesh_num() const
{
	return m_mesh_num;
}

inline float3 triangle_mesh::get_position(int index) const
{
	return m_mesh_position[index];
}

inline float3 triangle_mesh::get_scale(int index) const
{
	return m_mesh_scale[index];
}

inline float3 triangle_mesh::get_rotate(int index) const
{
	return m_mesh_rotate[index];
}

inline float3 triangle_mesh::get_rotate_applied(int index) const
{
	return m_mesh_rotate_applied[index];
}

inline material triangle_mesh::get_material(int index) const
{
	return m_mat_device[index];
}

inline int triangle_mesh::get_triangle_num(int index) const
{
	return m_mesh_triangles_num[index];
}

inline int triangle_mesh::get_vertex_num(int index) const
{
	return m_mesh_vertices_num[index];
}

inline triangle* triangle_mesh::get_triangles_device() const
{
	return m_mesh_device;
}

inline bvh_node_device** triangle_mesh::get_bvh_node_device() const
{
	return m_mesh_bvh_transformed_device;
}

inline bool triangle_mesh::create_bvh_device_data()
{
	if (m_mesh_num == 0 || m_mesh_device == nullptr)
	{
		return false;
	}

	CUDA_CALL(cudaMallocManaged((void**)&m_mesh_bvh_initial_device, m_mesh_num * sizeof(bvh_node_device*)));
	CUDA_CALL(cudaMallocManaged((void**)&m_mesh_bvh_transformed_device, m_mesh_num * sizeof(bvh_node_device*)));

	for (auto index = 0; index < m_mesh_num; index++)
	{
		int triangle_start_index = 0;
		for (auto j = 0; j < index; j++)
		{
			triangle_start_index += m_mesh_triangles_num[j];
		}

		double time;
		bvh_node* root;
		printf("[Info]Constructing bvh for mesh %s on cpu...\n", m_mesh_name[index].c_str());
		TIME_COUNT_CALL_START();
		root = BVH_BUILD_METHOD build_bvh(m_mesh_device + triangle_start_index, m_mesh_triangles_num[index], triangle_start_index);
		TIME_COUNT_CALL_END(time);
		printf("[Info]Completed, time consuming: %.4f ms\n", time);

		printf("[Info]Copy bvh data for mesh %s to GPU...\n", m_mesh_name[index].c_str());
		TIME_COUNT_CALL_START();
		m_mesh_bvh_initial_device[index] = BVH_BUILD_METHOD build_bvh_device_data(root); 
		m_mesh_bvh_transformed_device[index] = m_mesh_bvh_initial_device[index] + m_mesh_bvh_initial_device[index]->next_node_index;
		TIME_COUNT_CALL_END(time);
		printf("[Info]Completed, time consuming: %.4f ms\n", time);

		BVH_BUILD_METHOD release_bvh(root);
	}

	return true;
}

inline void triangle_mesh::release_bvh_device_data()
{
	if (m_mesh_bvh_initial_device != nullptr)
	{
		for (auto index = 0; index < m_mesh_num; index++)
		{
			CUDA_CALL(cudaFree(m_mesh_bvh_initial_device[index]));
			m_mesh_bvh_initial_device[index] = nullptr;
			m_mesh_bvh_transformed_device[index] = nullptr;
		}
		CUDA_CALL(cudaFree(m_mesh_bvh_initial_device));
		m_mesh_bvh_initial_device = nullptr;
		m_mesh_bvh_transformed_device = nullptr;
	}
}

inline bool triangle_mesh::create_mesh_device_data()
{
	if (!m_is_loaded)
	{
		return false;
	}

	printf("[Info]Copy mesh data to gpu...\n");
	double time;
	TIME_COUNT_CALL_START();

	std::vector<material> materials(m_triangles.size());
	for (auto i = 0; i < m_mesh_material.size(); i++)
	{
		materials[i] = *(m_mesh_material[i]);
	}

	CUDA_CALL(cudaMallocManaged((void**)&m_mat_device, materials.size() * sizeof(material)));
	CUDA_CALL(cudaMemcpy(m_mat_device, materials.data(), materials.size() * sizeof(material), cudaMemcpyDefault));

	CUDA_CALL(cudaMallocManaged((void**)&m_mesh_device, m_triangles.size() * sizeof(triangle)));
	CUDA_CALL(cudaMemcpy(m_mesh_device, m_triangles.data(), m_triangles.size() * sizeof(triangle), cudaMemcpyDefault));
		
	for (auto index = 0; index < m_mesh_num; index++)
	{
		int triangle_start_index = 0;
		for (auto i = 0; i < index; i++)
		{
			triangle_start_index += m_mesh_triangles_num[i];
		}
		
		for (auto i = 0; i < m_mesh_triangles_num[index]; i++)
		{
			m_mesh_device[i + triangle_start_index].mat = m_mat_device + index;

			float3 vertex0 = m_triangles[i + triangle_start_index].vertex0;
			float3 vertex1 = m_triangles[i + triangle_start_index].vertex1;
			float3 vertex2 = m_triangles[i + triangle_start_index].vertex2;

			float3 normal0 = m_triangles[i + triangle_start_index].normal0;
			float3 normal1 = m_triangles[i + triangle_start_index].normal1;
			float3 normal2 = m_triangles[i + triangle_start_index].normal2;

			glm::vec4 vertex0_vec4 = glm::vec4(vertex0.x, vertex0.y, vertex0.z, 1.0f);
			glm::vec4 vertex1_vec4 = glm::vec4(vertex1.x, vertex1.y, vertex1.z, 1.0f);
			glm::vec4 vertex2_vec4 = glm::vec4(vertex2.x, vertex2.y, vertex2.z, 1.0f);

			glm::vec4 normal0_vec4 = glm::vec4(normal0.x, normal0.y, normal0.z, 0.0f);
			glm::vec4 normal1_vec4 = glm::vec4(normal1.x, normal1.y, normal1.z, 0.0f);
			glm::vec4 normal2_vec4 = glm::vec4(normal2.x, normal2.y, normal2.z, 0.0f);

			vertex0_vec4  = m_mesh_current_transform[index] * vertex0_vec4;
			vertex1_vec4  = m_mesh_current_transform[index] * vertex1_vec4;
			vertex2_vec4  = m_mesh_current_transform[index] * vertex2_vec4;

			glm::mat4 inverse_transpose_transform_mat = glm::transpose(glm::inverse(m_mesh_current_transform[index]));
			normal0_vec4 = inverse_transpose_transform_mat * normal0_vec4;
			normal1_vec4 = inverse_transpose_transform_mat * normal1_vec4;
			normal2_vec4 = inverse_transpose_transform_mat * normal2_vec4;

			m_mesh_device[i + triangle_start_index].vertex0 = make_float3(vertex0_vec4.x, vertex0_vec4.y, vertex0_vec4.z);
			m_mesh_device[i + triangle_start_index].vertex1 = make_float3(vertex1_vec4.x, vertex1_vec4.y, vertex1_vec4.z);
			m_mesh_device[i + triangle_start_index].vertex2 = make_float3(vertex2_vec4.x, vertex2_vec4.y, vertex2_vec4.z);
			m_mesh_device[i + triangle_start_index].normal0 = normalize(make_float3(normal0_vec4.x, normal0_vec4.y, normal0_vec4.z));
			m_mesh_device[i + triangle_start_index].normal1 = normalize(make_float3(normal1_vec4.x, normal1_vec4.y, normal1_vec4.z));
			m_mesh_device[i + triangle_start_index].normal2 = normalize(make_float3(normal2_vec4.x, normal2_vec4.y, normal2_vec4.z));
		}
	}
	
	TIME_COUNT_CALL_END(time);
	printf("[Info]Completed, time consuming: %.4f ms\n", time);

	return true;
}

inline void triangle_mesh::release_mesh_device_data()
{
	if (m_mat_device != nullptr)
	{
		CUDA_CALL(cudaFree(m_mat_device));
		m_mat_device = nullptr;
	}
	if (m_mesh_device != nullptr)
	{
		CUDA_CALL(cudaFree(m_mesh_device));
		m_mesh_device = nullptr;
	}
}

#endif // !__TRIANGLE_MESH__