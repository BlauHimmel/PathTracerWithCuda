#pragma once

#ifndef __TRIANGLE_MESH__
#define __TRIANGLE_MESH__

#include <cuda_runtime.h>
#include "lib\tiny_obj_loader\tiny_obj_loader.h"
#include "material.hpp"
#include "utilities.hpp"
#include <iostream>

struct triangle
{
	float3 vertex0;
	float3 vertex1;
	float3 vertex2;
	float3 normal;

	material mat;
};

class triangle_mesh
{
private:
	std::vector<triangle> m_triangles;

	std::vector<int> m_mesh_triangles_num;
	std::vector<int> m_mesh_vertices_num;

	std::vector<float3> m_mesh_position;
	std::vector<material> m_mesh_material;

	int m_mesh_num = 0;

	bool m_is_loaded = false;

	triangle* m_mesh_device = nullptr;

public:
	bool load_obj(const std::string& filename);
	void unload_obj();

	void set_position(int index, const float3& position);
	void set_material(int index, const material& mat);

	int get_mesh_num() const;
	float3 get_position(int index) const;
	material get_material(int index) const;
	int get_triangle_num(int index) const;
	int get_vertex_num(int index) const;
	std::vector<triangle> get_triangles() const;

	triangle* create_mesh_device_data();
	void release_mesh_device_data();
	
};

//TODO:COULD LOAD MTL HERE(OR TEXTURE)
inline bool triangle_mesh::load_obj(const std::string& filename)
{
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	tinyobj::attrib_t attrib; 
	std::string error;

	int triangle_num = 0;

	std::cout << "[Info]Loading file " << filename << "...." << std::endl;

	bool is_success = tinyobj::LoadObj(&attrib, &shapes, &materials, &error, filename.c_str());

	if (!error.empty())
	{
		std::cout << "[TinyObj]\n" << error.substr(0, error.find_last_of('\n')) << std::endl;
	}

	if (!is_success)
	{
		std::cout << "[Info]Load file " << filename << "failed." << std::endl;
		return false;
	}

	for (auto i = 0; i < shapes.size(); i++)
	{
		for (auto num : shapes[i].mesh.num_face_vertices)
		{
			if (num != 3)
			{
				std::cout << "[Error]" <<  filename << " is not a triangle mesh." << std::endl;
				return false;
			}
		}

		for (auto f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++)
		{
			float3 vertex[3];
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

			triangle triangle;
 			triangle.vertex0 = vertex[0];
			triangle.vertex1 = vertex[1];
			triangle.vertex2 = vertex[2];
			triangle.normal = normalize(cross(vertex[1] - vertex[0], vertex[2] - vertex[0]));
			triangle.mat = get_default_material();
			m_triangles.push_back(triangle);

			triangle_num++;
		}
	}

	m_is_loaded = true;
	m_mesh_num++;

	int vertices_num = static_cast<int>(attrib.vertices.size() / 3);

	m_mesh_triangles_num.push_back(triangle_num);
	m_mesh_vertices_num.push_back(vertices_num);
	m_mesh_position.push_back(make_float3(0.0f, 0.0f, 0.0f));
	m_mesh_material.push_back(get_default_material());

	std::cout << "[Info]Load file " << filename << " succeeded. vertices : " << vertices_num << std::endl;

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
	m_mesh_material.clear();
	m_mesh_material.shrink_to_fit();

	m_is_loaded = false;
}

inline void triangle_mesh::set_position(int index, const float3& position)
{
	int triangle_start_index = 0;
	for (auto i = 0; i < index; i++)
	{
		triangle_start_index += m_mesh_triangles_num[i];
	}

	for (auto i = 0; i < m_mesh_triangles_num[index]; i++)
	{
		m_triangles[i + triangle_start_index].vertex0 += position;
		m_triangles[i + triangle_start_index].vertex1 += position;
		m_triangles[i + triangle_start_index].vertex2 += position;
	}

	m_mesh_position[index] = position;
}

inline void triangle_mesh::set_material(int index, const material& mat)
{
	int triangle_start_index = 0;
	for (auto i = 0; i < index; i++)
	{
		triangle_start_index += m_mesh_triangles_num[i];
	}

	for (auto i = 0; i < m_mesh_triangles_num[index]; i++)
	{
		m_triangles[i + triangle_start_index].mat = mat;
	}

	m_mesh_material[index] = mat;
}

inline int triangle_mesh::get_mesh_num() const
{
	return m_mesh_num;
}

inline float3 triangle_mesh::get_position(int index) const
{
	return m_mesh_position[index];
}

inline material triangle_mesh::get_material(int index) const
{
	return m_mesh_material[index];
}

inline int triangle_mesh::get_triangle_num(int index) const
{
	return m_mesh_triangles_num[index];
}

inline int triangle_mesh::get_vertex_num(int index) const
{
	return m_mesh_vertices_num[index];
}

inline std::vector<triangle> triangle_mesh::get_triangles() const
{
	return m_triangles;
}

inline triangle* triangle_mesh::create_mesh_device_data()
{
	if (!m_is_loaded)
	{
		return nullptr;
	}

	CUDA_CALL(cudaMalloc((void**)&m_mesh_device, m_triangles.size() * sizeof(triangle)));
	CUDA_CALL(cudaMemcpy(m_mesh_device, m_triangles.data(), m_triangles.size() * sizeof(triangle), cudaMemcpyHostToDevice));
	
	return m_mesh_device;
}

inline void triangle_mesh::release_mesh_device_data()
{
	if (m_mesh_device != nullptr)
	{
		CUDA_CALL(cudaFree(m_mesh_device));
		m_mesh_device = nullptr;
	}
}

#endif // !__TRIANGLE_MESH__