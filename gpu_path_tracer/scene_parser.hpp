#pragma once

#ifndef __SCENE_PARSER__
#define __SCENE_PARSER__

#include "lib\json\json.hpp"

#include "sphere.hpp"
#include "triangle_mesh.hpp"
#include "cube_map.hpp"
#include "material.hpp"
#include "bvh.hpp"

#include <fstream>

#define TOKEN_OBJECT_SPHERE "Sphere"
#define TOKEN_OBJECT_SPHERE_CENTER "Center"
#define TOKEN_OBJECT_SPHERE_RADIUS "Radius"
#define TOKEN_OBJECT_SPHERE_MATERIAL "Material"

#define TOKEN_OBJECT_MESH "Mesh"
#define TOKEN_OBJECT_MESH_PATH "Path"
#define TOKEN_OBJECT_MESH_MATERIAL "Material"
#define TOKEN_OBJECT_MESH_POSITION "Position"

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
	{
		"Material" : "XXXX",							-- name of material(user declared or built-in material)
		"Path" : "XXXX\\YYYY",
		"Position" : "0.0 0.0 0.0"
	}
}
*/


#define CHECK_PROPERTY(Category, Property, Token)\
if (Property.is_null())\
{\
	std::cout << "[Error]" << #Category << " property <" << Token << "> not defined!" << std::endl;\
	return false;\
}\

class scene_parser
{
private:
	nlohmann::json json_parser;

	std::map<std::string, material> m_materials;

	cube_map_loader m_cube_map_loader;
	cube_map* m_cube_map_device = nullptr;

	triangle_mesh m_triangle_mesh;
	int m_triangle_num = 0;
	int m_vertices_num = 0;
	triangle* m_triangles_device = nullptr;

	bvh_node_device* m_bvh_nodes_device = nullptr;

	sphere* m_spheres = nullptr;
	int m_sphere_num = 0;
	sphere* m_spheres_device = nullptr;

	bool m_is_loaded = false;

public:
	~scene_parser();

	bool load_scene(const std::string& filename);
	void unload_scene();

	void create_scene_data_device();
	void release_scene_data_device();

	cube_map* get_cube_map_device_ptr();
	triangle* get_triangles_device_ptr();
	bvh_node_device* get_bvh_node_device_ptr();
	sphere* get_sphere_device_ptr();
	int get_mesh_triangle_num();
	int get_mesh_vertices_num();
	int get_sphere_num();
	float3 get_mesh_position();
	material get_mesh_material();
	sphere get_sphere(int index);

	void set_sphere(int index, const sphere& sphere);
	void set_mesh_material(const material& material);

	void update_sphere_device_data();
	void update_triangles_device_data();

private:
	void init_default_material();

	float3 parse_float3(std::string text);
	float parse_float(std::string text);
	bool parse_bool(std::string text);
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

	init_default_material();

	std::ifstream scene_file(filename, std::ios::in);
	scene_file >> json_parser;

	std::vector<std::string> cubemap_pathes(6);
	std::map<std::string, material> materials;
	std::vector<sphere> spheres;
	std::vector<std::string> spheres_mat;
	std::string mesh_path;
	std::string mesh_mat;
	float3 mesh_position;

	//Step.1 Load data from file

	auto sphere_object = json_parser[TOKEN_OBJECT_SPHERE];
	auto mesh_object = json_parser[TOKEN_OBJECT_MESH];
	auto background_object = json_parser[TOKEN_BACKGROUND];
	auto material_object = json_parser[TOKEN_MATERIAL];

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

			material mat = material{
				parse_float3(diffuse_str),
				parse_float3(emission_str),
				parse_float3(specular_str),
				parse_bool(transparent_str),
				parse_float(roughness_str),
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
			sphere.radius = parse_float(radius_str);
			sphere.mat = get_default_material();

			spheres.push_back(sphere);
			spheres_mat.push_back(material_str);
		}
	}

	//Mesh
	if (!mesh_object.is_null() && mesh_object.size() > 0)
	{
		auto path = mesh_object[TOKEN_OBJECT_MESH_PATH];
		auto material = mesh_object[TOKEN_OBJECT_MESH_MATERIAL];
		auto position = mesh_object[TOKEN_OBJECT_MESH_POSITION];

		CHECK_PROPERTY(Mesh, path, TOKEN_OBJECT_MESH_PATH);
		CHECK_PROPERTY(Mesh, material, TOKEN_OBJECT_MESH_MATERIAL);
		CHECK_PROPERTY(Mesh, position, TOKEN_OBJECT_MESH_MATERIAL);

		std::string path_str = path;
		std::string material_str = material;
		std::string position_str = position;

		mesh_path = path_str;
		mesh_mat = material_str;
		mesh_position = parse_float3(position_str);
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
	for (auto mat : materials)
	{
		if (m_materials.find(mat.first) != m_materials.end())
		{
			std::cout << "[Warning]Built-in materail <" << mat.first << "> is overlapped!" << std::endl;
		}

		m_materials[mat.first] = mat.second;
	}

	for (auto sphere_mat : spheres_mat)
	{
		if (m_materials.find(sphere_mat) == m_materials.end())
		{
			std::cout << "[Error]Material <" << sphere_mat << "> not found!" << std::endl;
			error = true;
		}
	}

	if (m_materials.find(mesh_mat) == m_materials.end())
	{
		std::cout << "[Error]Material <" << mesh_mat << "> not found!" << std::endl;
		error = true;
	}
	
	if (error)
	{
		m_materials.clear();
		m_cube_map_loader.unload_data();
		return false;
	}

	//Mesh
	error = !m_triangle_mesh.load_obj(mesh_path);

	if (error)
	{
		m_materials.clear();
		m_cube_map_loader.unload_data();
		m_triangle_mesh.unload_obj();
		return false;
	}

	m_triangle_mesh.set_material(m_materials[mesh_mat]);
	m_triangle_mesh.set_position(mesh_position);
	m_triangle_num = m_triangle_mesh.get_triangle_num();
	m_vertices_num = m_triangle_mesh.get_vertex_num();

	//Sphere
	for (auto i = 0; i < spheres.size(); i++)
	{
		spheres[i].mat = m_materials[spheres_mat[i]];
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
		m_vertices_num = 0;
		m_triangle_num = 0;
		m_sphere_num = 0;
	}
}

inline void scene_parser::create_scene_data_device()
{
	if (!m_is_loaded)
	{
		return;
	}

	double time;

	printf("[Info]Copy background data to gpu...\n");
	TIME_COUNT_CALL_START();
	m_cube_map_device = m_cube_map_loader.create_cube_device_data();
	TIME_COUNT_CALL_END(time);
	printf("[Info]Completed, time consuming: %.4f ms\n", time);

	printf("[Info]Copy sphere data to gpu...\n");
	TIME_COUNT_CALL_START();
	CUDA_CALL(cudaMalloc((void**)&m_spheres_device, m_sphere_num * sizeof(sphere)));
	CUDA_CALL(cudaMemcpy(m_spheres_device, m_spheres, m_sphere_num * sizeof(sphere), cudaMemcpyHostToDevice));
	TIME_COUNT_CALL_END(time);
	printf("[Info]Completed, time consuming: %.4f ms\n", time);

	printf("[Info]Copy triangle data to gpu...\n");
	TIME_COUNT_CALL_START();
	m_triangles_device = m_triangle_mesh.create_mesh_device_data();
	TIME_COUNT_CALL_END(time);
	printf("[Info]Completed, time consuming: %.4f ms\n", time);

	bvh_node* root;
	printf("[Info]Constructing bvh on cpu...\n");
	TIME_COUNT_CALL_START();
	root = build_bvh(m_triangle_mesh.get_triangles());
	TIME_COUNT_CALL_END(time);
	printf("[Info]Completed, time consuming: %.4f ms\n", time);

	printf("[Info]Copy bvh data to GPU...\n");
	TIME_COUNT_CALL_START();
	m_bvh_nodes_device = build_bvh_device_data(root);
	TIME_COUNT_CALL_END(time);
	printf("[Info]Completed, time consuming: %.4f ms\n", time);
}

inline void scene_parser::release_scene_data_device()
{
	m_cube_map_loader.release_cube_device_data();
	m_cube_map_device = nullptr;
	m_triangle_mesh.release_mesh_device_data();
	m_triangles_device = nullptr;
	if (m_spheres_device != nullptr)
	{
		CUDA_CALL(cudaFree(m_spheres_device));
		m_spheres_device = nullptr;
	}
}

inline cube_map* scene_parser::get_cube_map_device_ptr()
{
	return m_cube_map_device;
}

inline triangle* scene_parser::get_triangles_device_ptr()
{
	return m_triangles_device;
}

inline bvh_node_device* scene_parser::get_bvh_node_device_ptr()
{
	return m_bvh_nodes_device;
}

inline sphere* scene_parser::get_sphere_device_ptr()
{
	return m_spheres_device;
}

inline int scene_parser::get_mesh_triangle_num()
{
	return m_triangle_num;
}

inline int scene_parser::get_mesh_vertices_num()
{
	return m_vertices_num;
}

inline int scene_parser::get_sphere_num()
{
	return m_sphere_num;
}

inline float3 scene_parser::get_mesh_position()
{
	return m_triangle_mesh.get_position();
}

inline material scene_parser::get_mesh_material()
{
	return m_triangle_mesh.get_material();
}

inline sphere scene_parser::get_sphere(int index)
{
	return m_spheres[index];
}

inline void scene_parser::set_sphere(int index, const sphere& sphere)
{
	m_spheres[index] = sphere;
}

inline void scene_parser::set_mesh_material(const material& material)
{
	m_triangle_mesh.set_material(material);
}

inline void scene_parser::update_sphere_device_data()
{
	if (m_spheres_device == nullptr)
	{
		return;
	}

	CUDA_CALL(cudaFree(m_spheres_device));
	CUDA_CALL(cudaMalloc((void**)&m_spheres_device, m_sphere_num * sizeof(sphere)));
	CUDA_CALL(cudaMemcpy(m_spheres_device, m_spheres, m_sphere_num * sizeof(sphere), cudaMemcpyHostToDevice));
}

inline void scene_parser::update_triangles_device_data()
{
	m_triangle_mesh.release_mesh_device_data();
	m_triangles_device = m_triangle_mesh.create_mesh_device_data();
}

void scene_parser::init_default_material()
{
	m_materials.clear();

	m_materials.insert(std::make_pair("titanium",		material_data::metal::titanium()));
	m_materials.insert(std::make_pair("chromium",		material_data::metal::chromium()));
	m_materials.insert(std::make_pair("iron",			material_data::metal::iron()));
	m_materials.insert(std::make_pair("nickel",			material_data::metal::nickel()));
	m_materials.insert(std::make_pair("platinum",		material_data::metal::platinum()));
	m_materials.insert(std::make_pair("copper",			material_data::metal::copper()));
	m_materials.insert(std::make_pair("palladium",		material_data::metal::palladium()));
	m_materials.insert(std::make_pair("zinc",			material_data::metal::zinc()));
	m_materials.insert(std::make_pair("gold",			material_data::metal::gold()));
	m_materials.insert(std::make_pair("aluminum",		material_data::metal::aluminum()));
	m_materials.insert(std::make_pair("silver",			material_data::metal::silver()));

	m_materials.insert(std::make_pair("glass",			material_data::dielectric::glass()));
	m_materials.insert(std::make_pair("green_glass",	material_data::dielectric::green_glass()));
	m_materials.insert(std::make_pair("diamond",		material_data::dielectric::diamond()));
	m_materials.insert(std::make_pair("red",			material_data::dielectric::red()));
	m_materials.insert(std::make_pair("green",			material_data::dielectric::green()));
	m_materials.insert(std::make_pair("orange",			material_data::dielectric::orange()));
	m_materials.insert(std::make_pair("purple",			material_data::dielectric::purple()));
	m_materials.insert(std::make_pair("blue",			material_data::dielectric::blue()));
	m_materials.insert(std::make_pair("marble",			material_data::dielectric::marble()));
	m_materials.insert(std::make_pair("something_blue", material_data::dielectric::something_blue()));
	m_materials.insert(std::make_pair("something_red",	material_data::dielectric::something_red()));

	m_materials.insert(std::make_pair("light",			material_data::dielectric::light()));
}

inline float3 scene_parser::parse_float3(std::string text)
{
	std::istringstream stream(text);
	float x, y, z;
	stream >> x >> y >> z;
	return make_float3(x, y, z);
}

inline float scene_parser::parse_float(std::string text)
{
	std::istringstream stream(text);
	float value;
	stream >> value;
	return value;
}

inline bool scene_parser::parse_bool(std::string text)
{
	std::istringstream stream(text);
	bool value;
	stream >> value;
	return value;
}

#endif // !__SCENE_PARSER__