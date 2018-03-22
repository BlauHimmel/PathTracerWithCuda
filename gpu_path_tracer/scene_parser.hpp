#pragma once

#ifndef __SCENE_PARSER__
#define __SCENE_PARSER__

#include "lib\json\json.hpp"

#include "sphere.hpp"
#include "triangle_mesh.hpp"
#include "cube_map.hpp"
#include "material.hpp"
#include "bvh.h"

#include <exception>
#include <fstream>

#define TOKEN_OBJECT_SPHERE "Sphere"
#define TOKEN_OBJECT_SPHERE_CENTER "Center"
#define TOKEN_OBJECT_SPHERE_RADIUS "Radius"
#define TOKEN_OBJECT_SPHERE_MATERIAL "Material"

#define TOKEN_OBJECT_MESH "Mesh"
#define TOKEN_OBJECT_MESH_PATH "Path"
#define TOKEN_OBJECT_MESH_MATERIAL "Material"
#define TOKEN_OBJECT_MESH_POSITION "Position"
#define TOKEN_OBJECT_MESH_SCALE "Scale"

#define TOKEN_BACKGROUND "Background"
#define TOKEN_BACKGROUND_CUBE_MAP_ROOT_PATH "Path"
#define TOKEN_BACKGROUND_CUBE_MAP_NAME "Name"

#define TOKEN_MATERIAL "Material"
#define TOKEN_MATERIAL_NAME "Name"
#define TOKEN_MATERIAL_DIFFUSE "Diffuse"
#define TOKEN_MATERIAL_EMISSION "Emission"
#define TOKEN_MATERIAL_SPECULAR "Specular"
#define TOKEN_MATERIAL_TRANSPARENT "Transparent"
#define TOKEN_MATERIAL_ROUGHNESS "Roughness"
#define TOKEN_MATERIAL_REFRACTION_INDEX "RefractionIndex"
#define TOKEN_MATERIAL_EXTINCTION_COEF "ExtinctionCoef"
#define TOKEN_MATERIAL_ABSORPTION_COEF "AbsorptionCoef"
#define TOKEN_MATERIAL_REDUCED_SCATTERING_COEF "ReducedScatteringCoef"

/*
{
	"Background" : 
	{
		"Name" : "XXXX",
		"Path" : "XXXX\\YYYY\\"
	},

	-- built-in material --
	-- titanium, chromium, iron, nickel, platinum, copper, palladium, zinc, gold, aluminum, silver --
	-- glass, green_glass, red, green, orange, purple, blue, marble, something_blue, something_red --
	-- light --

	-- optional --
	"Material" :
	[
		{
			"Name" : "XXXX",
			"Diffuse" : "0.0 0.0 0.0",					-- each component >= 0.0 --
			"Emission" : "0.0 0.0 0.0",					-- each component >= 0.0 --
			"Specular" : "0.0 0.0 0.0",					-- each component >= 0.0 --
			"Transparent" : "true",						-- true or false --
			"Roughness" : "0.0",						-- from 0.0 to 1.0 --
			"RefractionIndex" : "0.0",					-- any value >= 0.0 --
			"ExtinctionCoef" : "0.0",					-- any value >= 0.0 --
			"AbsorptionCoef" : "0.0 0.0 0.0",			-- each component >= 0.0 --
			"ReducedScatteringCoef" : "0.0 0.0 0.0"		-- each component >= 0.0 --
		},
		...
	],

	-- optional --
	"Sphere" :
	[
		{
			"Material" : "XXXX",						-- name of material(user declared or built-in material)
			"Center" : "0.0 0.0 0.0",
			"Radius" : "0.0"							-- any value >= 0.0 --
		},
		...
	],

	-- optional --
	"Mesh" :
	[
		{
			"Material" : "XXXX",							-- name of material(user declared or built-in material)
			"Path" : "XXXX\\YYYY",
			"Position" : "0.0 0.0 0.0",
			"Scale" : "1.0 1.0 1.0"
		},
		...
	]
}
*/

struct triangle;

class scene_parser
{
private:
	nlohmann::json m_json_parser;

	cube_map_loader m_cube_map_loader;

	triangle_mesh m_triangle_mesh;

	sphere* m_spheres = nullptr;
	int m_sphere_num = 0;
	sphere* m_spheres_device = nullptr;

	bool m_is_loaded = false;

public:
	~scene_parser();

	bool load_scene(const std::string& filename);
	void unload_scene();

	bool create_scene_data_device();
	void release_scene_data_device();

	cube_map* get_cube_map_device_ptr();
	triangle* get_triangles_device_ptr();
	bvh_node_device** get_bvh_node_device_ptr();
	sphere* get_sphere_device_ptr();

	int get_mesh_num() const;
	int get_triangles_num() const;
	int get_mesh_triangle_num(int index) const;
	int get_mesh_vertices_num(int index) const;
	int get_sphere_num() const;
	float3 get_mesh_position(int index) const;
	material get_mesh_material(int index) const;
	float3 get_mesh_scale(int index) const;
	sphere get_sphere(int index) const;

	void set_sphere_device(int index, const sphere& sphere);
	void set_mesh_material_device(int index, const material& material);
	void set_mesh_transform_device(
		int index,
		const float3& position,
		const float3& scale,
		std::function<void(const float3&, const float3&, const float3&, const float3&, bvh_node_device*)> bvh_update_function
	);

private:
	void init_default_material(std::map<std::string, material>& materials);

	float3 parse_float3(const std::string& text);
	float parse_float(const std::string& text);
	bool parse_bool(const std::string& text);
};

inline scene_parser::~scene_parser()
{
	release_scene_data_device();
	unload_scene();
}

inline bool scene_parser::load_scene(const std::string& filename)
{
	if (m_is_loaded)
	{
		return false;
	}

	try
	{
		std::ifstream scene_file(filename, std::ios::in);
		scene_file >> m_json_parser;
	}
	catch (nlohmann::detail::parse_error error)
	{
		std::cout << "[ERROR]" << error.what() << std::endl;
		return false;
	}

	std::vector<std::string> cubemap_pathes(6);

	std::map<std::string, material> materials;

	std::vector<sphere> spheres;
	std::vector<std::string> spheres_mat;

	std::vector<std::string> meshes_path;
	std::vector<std::string> meshes_mat;
	std::vector<float3> meshes_position;
	std::vector<float3> meshes_scale;

	init_default_material(materials);

	//Step.1 Load data from file

	auto sphere_object = m_json_parser[TOKEN_OBJECT_SPHERE];
	auto mesh_object = m_json_parser[TOKEN_OBJECT_MESH];
	auto background_object = m_json_parser[TOKEN_BACKGROUND];
	auto material_object = m_json_parser[TOKEN_MATERIAL];

	//Background
	if (background_object.is_null())
	{
		std::cout << "[Error]Background not defined!" << std::endl;
		return false;
	}
	else if (background_object.is_array())
	{
		std::cout << "[Error]Background can not be array!" << std::endl;
		return false;
	}
	else
	{
		auto name = background_object[TOKEN_BACKGROUND_CUBE_MAP_NAME];
		auto root_path = background_object[TOKEN_BACKGROUND_CUBE_MAP_ROOT_PATH];
		
		CHECK_PROPERTY(Background, name, TOKEN_BACKGROUND_CUBE_MAP_NAME);
		CHECK_PROPERTY(Background, root_path, TOKEN_BACKGROUND_CUBE_MAP_ROOT_PATH);

		std::string name_str = name;
		std::string root_path_str = root_path;
		std::string path = root_path_str + name_str;

		cubemap_pathes[0] = path + "\\xpos.bmp";
		cubemap_pathes[1] = path + "\\xneg.bmp";
		cubemap_pathes[2] = path + "\\ypos.bmp";
		cubemap_pathes[3] = path + "\\yneg.bmp";
		cubemap_pathes[4] = path + "\\zpos.bmp";
		cubemap_pathes[5] = path + "\\zneg.bmp";
	}

	//Material
	if (!material_object.is_null())
	{
		for (auto material_element : material_object)
		{
			auto name = material_element[TOKEN_MATERIAL_NAME];
			auto diffuse = material_element[TOKEN_MATERIAL_DIFFUSE];
			auto emission = material_element[TOKEN_MATERIAL_EMISSION];
			auto specular = material_element[TOKEN_MATERIAL_SPECULAR];
			auto transparent = material_element[TOKEN_MATERIAL_TRANSPARENT];
			auto roughness = material_element[TOKEN_MATERIAL_ROUGHNESS];
			auto refraction_index = material_element[TOKEN_MATERIAL_REFRACTION_INDEX];
			auto extinction_coef = material_element[TOKEN_MATERIAL_EXTINCTION_COEF];
			auto absorption_coef = material_element[TOKEN_MATERIAL_ABSORPTION_COEF];
			auto reduced_scattering_coef = material_element[TOKEN_MATERIAL_REDUCED_SCATTERING_COEF];

			CHECK_PROPERTY(Material, name, TOKEN_MATERIAL_NAME);
			CHECK_PROPERTY(Material, diffuse, TOKEN_MATERIAL_DIFFUSE);
			CHECK_PROPERTY(Material, emission, TOKEN_MATERIAL_EMISSION);
			CHECK_PROPERTY(Material, specular, TOKEN_MATERIAL_SPECULAR);
			CHECK_PROPERTY(Material, transparent, TOKEN_MATERIAL_TRANSPARENT);
			CHECK_PROPERTY(Material, roughness, TOKEN_MATERIAL_ROUGHNESS);
			CHECK_PROPERTY(Material, refraction_index, TOKEN_MATERIAL_REFRACTION_INDEX);
			CHECK_PROPERTY(Material, extinction_coef, TOKEN_MATERIAL_EXTINCTION_COEF);
			CHECK_PROPERTY(Material, absorption_coef, TOKEN_MATERIAL_ABSORPTION_COEF);
			CHECK_PROPERTY(Material, reduced_scattering_coef, TOKEN_MATERIAL_REDUCED_SCATTERING_COEF);
			
			std::string name_str = name;
			std::string diffuse_str = diffuse;
			std::string emission_str = emission;
			std::string specular_str = specular;
			std::string transparent_str = transparent;
			std::string roughness_str = roughness;
			std::string refraction_index_str = refraction_index;
			std::string extinction_coef_str = extinction_coef;
			std::string absorption_coef_str = absorption_coef;
			std::string reduced_scattering_coef_str = reduced_scattering_coef;

			if (materials.find(name_str) != materials.end())
			{
				std::cout << "[Warning]Built-in materail <" << name_str << "> is overlapped!" << std::endl;
			}

			material mat = material{
				parse_float3(diffuse_str),
				parse_float3(emission_str),
				parse_float3(specular_str),
				parse_bool(transparent_str),
				clamp(parse_float(roughness_str), 0.0f, 1.0f),
				{
					parse_float(refraction_index_str),
					parse_float(extinction_coef_str),
					{
						parse_float3(absorption_coef_str),
						parse_float3(reduced_scattering_coef_str)
					}
				}
			};


			materials[name_str] = mat;
		}
	}

	//Sphere
	if (!sphere_object.is_null())
	{
		for (auto sphere_element : sphere_object)
		{
			auto center = sphere_element[TOKEN_OBJECT_SPHERE_CENTER];
			auto radius = sphere_element[TOKEN_OBJECT_SPHERE_RADIUS];
			auto material = sphere_element[TOKEN_OBJECT_SPHERE_MATERIAL];

			CHECK_PROPERTY(Sphere, center, TOKEN_OBJECT_SPHERE_CENTER);
			CHECK_PROPERTY(Sphere, radius, TOKEN_OBJECT_SPHERE_RADIUS);
			CHECK_PROPERTY(Sphere, material, TOKEN_OBJECT_SPHERE_MATERIAL);

			std::string center_str = center;
			std::string radius_str = radius;
			std::string material_str = material;

			sphere sphere;
			sphere.center = parse_float3(center_str);
			sphere.radius = clamp(parse_float(radius_str), 0.0f, INFINITY);
			sphere.mat = get_default_material();

			spheres.push_back(sphere);
			spheres_mat.push_back(material_str);
		}
	}

	//Mesh
	if (!mesh_object.is_null())
	{
		for (auto mesh_element : mesh_object)
		{
			auto path = mesh_element[TOKEN_OBJECT_MESH_PATH];
			auto material = mesh_element[TOKEN_OBJECT_MESH_MATERIAL];
			auto position = mesh_element[TOKEN_OBJECT_MESH_POSITION];
			auto scale = mesh_element[TOKEN_OBJECT_MESH_SCALE];

			CHECK_PROPERTY(Mesh, path, TOKEN_OBJECT_MESH_PATH);
			CHECK_PROPERTY(Mesh, material, TOKEN_OBJECT_MESH_MATERIAL);
			CHECK_PROPERTY(Mesh, position, TOKEN_OBJECT_MESH_MATERIAL);
			CHECK_PROPERTY(Mesh, scale, TOKEN_OBJECT_MESH_SCALE);

			std::string path_str = path;
			std::string material_str = material;
			std::string position_str = position;
			std::string scale_str = scale;

			meshes_path.push_back(path_str);
			meshes_mat.push_back(material_str);
			meshes_position.push_back(parse_float3(position_str));
			meshes_scale.push_back(clamp(parse_float3(scale_str), make_float3(0.0f, 0.0f, 0.0f), make_float3(INFINITY, INFINITY, INFINITY)));
		}
	}

	//Step.2 Check data and create scene on cpu
	bool error = false;

	//Background
	error = !m_cube_map_loader.load_data(
		cubemap_pathes[0],
		cubemap_pathes[1],
		cubemap_pathes[2],
		cubemap_pathes[3],
		cubemap_pathes[4],
		cubemap_pathes[5]
	);

	if (error)
	{
		m_cube_map_loader.unload_data();
		std::cout << "[Error]Background load fail, please check the <Path> and <Name>!" << std::endl;
		return false;
	}

	//Material
	for (auto sphere_mat : spheres_mat)
	{
		if (materials.find(sphere_mat) == materials.end())
		{
			std::cout << "[Error]Material <" << sphere_mat << "> not found!" << std::endl;
			error = true;
		}
	}

	for (auto mesh_mat : meshes_mat)
	{
		if (materials.find(mesh_mat) == materials.end())
		{
			std::cout << "[Error]Material <" << mesh_mat << "> not found!" << std::endl;
			error = true;
		}
	}

	if (error)
	{
		m_cube_map_loader.unload_data();
		return false;
	}

	//Mesh
	for (auto i = 0; i < meshes_path.size(); i++)
	{
		error = !m_triangle_mesh.load_obj(meshes_path[i], meshes_position[i], meshes_scale[i], copy_material(materials[meshes_mat[i]]));

		if (error)
		{
			m_cube_map_loader.unload_data();
			m_triangle_mesh.unload_obj();
			return false;
		}
	}

	//Sphere
	for (auto i = 0; i < spheres.size(); i++)
	{
		spheres[i].mat = materials[spheres_mat[i]];
	}

	m_sphere_num = static_cast<int>(spheres.size());
	m_spheres = new sphere[m_sphere_num];
	memcpy(m_spheres, spheres.data(), m_sphere_num * sizeof(sphere));

	m_is_loaded = true;

	return true;
}

inline void scene_parser::unload_scene()
{
	if (m_is_loaded)
	{
		m_is_loaded = false;
		SAFE_DELETE_ARRAY(m_spheres);
		m_cube_map_loader.unload_data();
		m_triangle_mesh.unload_obj();
		m_sphere_num = 0;
	}
}

inline bool scene_parser::create_scene_data_device()
{
	if (!m_is_loaded)
	{
		return false;
	}

	double time;

	printf("[Info]Copy background data to gpu...\n");
	TIME_COUNT_CALL_START();
	if (!m_cube_map_loader.create_cube_device_data())
	{
		m_cube_map_loader.release_cube_device_data();
		return false;
	}
	TIME_COUNT_CALL_END(time);
	printf("[Info]Completed, time consuming: %.4f ms\n", time);

	printf("[Info]Copy sphere data to gpu...\n");
	TIME_COUNT_CALL_START();
	CUDA_CALL(cudaMallocManaged((void**)&m_spheres_device, m_sphere_num * sizeof(sphere)));
	CUDA_CALL(cudaMemcpy(m_spheres_device, m_spheres, m_sphere_num * sizeof(sphere), cudaMemcpyDefault));
	TIME_COUNT_CALL_END(time);
	printf("[Info]Completed, time consuming: %.4f ms\n", time);

	printf("[Info]Copy triangle data to gpu...\n");
	TIME_COUNT_CALL_START();
	if (!m_triangle_mesh.create_mesh_device_data())
	{
		m_cube_map_loader.release_cube_device_data();
		m_triangle_mesh.release_mesh_device_data();
		return false;
	}
	TIME_COUNT_CALL_END(time);
	printf("[Info]Completed, time consuming: %.4f ms\n", time);

	if (m_triangle_mesh.get_triangles_device() != nullptr && m_cube_map_loader.get_cube_map_device() != nullptr && m_spheres_device != nullptr)
	{
		m_triangle_mesh.create_bvh_device_data();
		return true;
	}
	
	return false;
}

inline void scene_parser::release_scene_data_device()
{
	m_cube_map_loader.release_cube_device_data();
	m_triangle_mesh.release_mesh_device_data();
	m_triangle_mesh.release_bvh_device_data();
	if (m_spheres_device != nullptr)
	{
		CUDA_CALL(cudaFree(m_spheres_device));
		m_spheres_device = nullptr;
	}
}

inline cube_map* scene_parser::get_cube_map_device_ptr()
{
	return m_cube_map_loader.get_cube_map_device();
}

inline triangle* scene_parser::get_triangles_device_ptr()
{
	return m_triangle_mesh.get_triangles_device();
}

inline bvh_node_device** scene_parser::get_bvh_node_device_ptr()
{
	return m_triangle_mesh.get_bvh_node_device();
}

inline sphere* scene_parser::get_sphere_device_ptr()
{
	return m_spheres_device;
}

inline int scene_parser::get_mesh_num() const
{
	return m_triangle_mesh.get_mesh_num();
}

inline int scene_parser::get_triangles_num() const
{
	return m_triangle_mesh.get_total_triangle_num();
}

inline int scene_parser::get_mesh_triangle_num(int index) const
{
	return m_triangle_mesh.get_triangle_num(index);
}

inline int scene_parser::get_mesh_vertices_num(int index) const
{
	return m_triangle_mesh.get_vertex_num(index);
}

inline int scene_parser::get_sphere_num() const
{
	return m_sphere_num;
}

inline float3 scene_parser::get_mesh_position(int index) const
{
	return m_triangle_mesh.get_position(index);
}

inline material scene_parser::get_mesh_material(int index) const
{
	return m_triangle_mesh.get_material(index);
}

inline float3 scene_parser::get_mesh_scale(int index) const
{
	return m_triangle_mesh.get_scale(index);
}

inline sphere scene_parser::get_sphere(int index) const
{
	return m_spheres_device[index];
}

inline void scene_parser::set_sphere_device(int index, const sphere& sphere)
{
	m_spheres_device[index] = sphere;
}

inline void scene_parser::set_mesh_material_device(int index, const material& material)
{
	m_triangle_mesh.set_material_device(index, material);
}

inline void scene_parser::set_mesh_transform_device(
	int index, 
	const float3& position,
	const float3& scale, 
	std::function<void(const float3&, const float3&, const float3&, const float3&, bvh_node_device*)> bvh_update_function
)
{
	m_triangle_mesh.set_transform_device(index, position, scale, bvh_update_function);
}

void scene_parser::init_default_material(std::map<std::string, material>& materials)
{
	materials.clear();

	materials.insert(std::make_pair("titanium",			material_data::metal::titanium()));
	materials.insert(std::make_pair("chromium",			material_data::metal::chromium()));
	materials.insert(std::make_pair("iron",				material_data::metal::iron()));
	materials.insert(std::make_pair("nickel",			material_data::metal::nickel()));
	materials.insert(std::make_pair("platinum",			material_data::metal::platinum()));
	materials.insert(std::make_pair("copper",			material_data::metal::copper()));
	materials.insert(std::make_pair("palladium",		material_data::metal::palladium()));
	materials.insert(std::make_pair("zinc",				material_data::metal::zinc()));
	materials.insert(std::make_pair("gold",				material_data::metal::gold()));
	materials.insert(std::make_pair("aluminum",			material_data::metal::aluminum()));
	materials.insert(std::make_pair("silver",			material_data::metal::silver()));

	materials.insert(std::make_pair("glass",			material_data::dielectric::glass()));
	materials.insert(std::make_pair("green_glass",		material_data::dielectric::green_glass()));
	materials.insert(std::make_pair("diamond",			material_data::dielectric::diamond()));
	materials.insert(std::make_pair("red",				material_data::dielectric::red()));
	materials.insert(std::make_pair("green",			material_data::dielectric::green()));
	materials.insert(std::make_pair("orange",			material_data::dielectric::orange()));
	materials.insert(std::make_pair("purple",			material_data::dielectric::purple()));
	materials.insert(std::make_pair("blue",				material_data::dielectric::blue()));
	materials.insert(std::make_pair("marble",			material_data::dielectric::marble()));
	materials.insert(std::make_pair("something_blue",	material_data::dielectric::something_blue()));
	materials.insert(std::make_pair("something_red",	material_data::dielectric::something_red()));

	materials.insert(std::make_pair("light",			material_data::dielectric::light()));
}

inline float3 scene_parser::parse_float3(const std::string& text)
{
	std::istringstream stream(text);
	float x, y, z;
	stream >> x >> y >> z;
	return make_float3(x, y, z);
}

inline float scene_parser::parse_float(const std::string& text)
{
	std::istringstream stream(text);
	float value;
	stream >> value;
	return value;
}

inline bool scene_parser::parse_bool(const std::string& text)
{
	if (text == "true")
	{
		return true;
	}
	else
	{
		return false;
	}
}

#endif // !__SCENE_PARSER__