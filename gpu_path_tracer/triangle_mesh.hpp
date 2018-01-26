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
	int m_triangle_num = 0;
	bool m_is_loaded = false;
	triangle* m_mesh_device = nullptr;

public:
	bool load_obj(const std::string& filename);
	void unload_obj();

	void set_position(const float3& position);
	void set_material(const material& mat);

	int get_triangle_num();
	std::vector<triangle> get_triangles();

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

	std::cout << "loading file " << filename << "...." << std::endl;

	bool is_success = tinyobj::LoadObj(&attrib, &shapes, &materials, &error, filename.c_str());

	if (!error.empty())
	{
		std::cout << "error: " << error;
	}

	if (!is_success)
	{
		std::cout << "load file " << filename << "failed." << std::endl;
		return false;
	}

	for (auto i = 0; i < shapes.size(); i++)
	{
		for (auto num : shapes[i].mesh.num_face_vertices)
		{
			if (num != 3)
			{
				std::cout << filename << " is not a triangle mesh." << std::endl;
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
		}
	}

	m_is_loaded = true;
	m_triangle_num = static_cast<int>(m_triangles.size());

	std::cout << "load file " << filename << " succeeded." << std::endl;

	return true;
}

inline void triangle_mesh::unload_obj()
{
	m_triangles.clear();
	m_triangles.shrink_to_fit();
	m_triangle_num = 0;
	m_is_loaded = false;
}

inline void triangle_mesh::set_position(const float3& position)
{
	for (auto i = 0; i < m_triangles.size(); i++)
	{
		m_triangles[i].vertex0 += position;
		m_triangles[i].vertex1 += position;
		m_triangles[i].vertex2 += position;
	}
}

inline void triangle_mesh::set_material(const material& mat)
{
	for (auto i = 0; i < m_triangles.size(); i++)
	{
		m_triangles[i].mat = mat;
	}
}

inline int triangle_mesh::get_triangle_num()
{
	return m_triangle_num;
}

inline std::vector<triangle> triangle_mesh::get_triangles()
{
	return m_triangles;
}

inline triangle* triangle_mesh::create_mesh_device_data()
{
	if (!m_is_loaded)
	{
		return nullptr;
	}

	CUDA_CALL(cudaMalloc((void**)&m_mesh_device, m_triangle_num * sizeof(triangle)));
	CUDA_CALL(cudaMemcpy(m_mesh_device, m_triangles.data(), m_triangle_num * sizeof(triangle), cudaMemcpyHostToDevice));
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