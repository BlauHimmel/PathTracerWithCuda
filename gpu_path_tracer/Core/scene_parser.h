#pragma once

#ifndef __SCENE_PARSER__
#define __SCENE_PARSER__

#include "lib\json\json.hpp"
#include "lib\glm\glm.hpp"

#include "Core\sphere.h"
#include "Core\triangle.h"
#include "Core\triangle_mesh.h"
#include "Core\cube_map_loader.h"
#include "Core\material.h"
#include "Others\image_loader.h"
#include "Bvh\bvh.h"

#include <exception>
#include <fstream>
#include <sstream>
#include <Windows.h>
#include <io.h>

#define TOKEN_OBJECT_SPHERE "Sphere"
#define TOKEN_OBJECT_SPHERE_CENTER "Center"
#define TOKEN_OBJECT_SPHERE_RADIUS "Radius"
#define TOKEN_OBJECT_SPHERE_MATERIAL "Material"

#define TOKEN_OBJECT_MESH "Mesh"
#define TOKEN_OBJECT_MESH_PATH "Path"
#define TOKEN_OBJECT_MESH_MATERIAL "Material"
#define TOKEN_OBJECT_MESH_POSITION "Position"
#define TOKEN_OBJECT_MESH_SCALE "Scale"
#define TOKEN_OBJECT_MESH_ROTATE "Rotate"

#define TOKEN_BACKGROUND "Background"
#define TOKEN_BACKGROUND_CUBE_MAP_ROOT_PATH "Path"
#define TOKEN_BACKGROUND_CUBE_MAP_NAME "Name"

#define TOKEN_TEXTURE "Texture"

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
#define TOKEN_MATERIAL_DIFFUSE_TEXTURE_ID "DiffuseTextureId"
#define TOKEN_MATERIAL_SPECULAR_TEXTURE_ID "SpecularTextureId"

/*
{
	Note: value check is imperfect, please check it when error occurred!
	"Background" : 
	{
		"Name" : "XXXX",								-- name of cube map(in this folder, your image must be named as [xpos,xneg,ypos,yneg,zpos,zneg])
		"Path" : "XXXX\\YYYY\\"							-- root folder of cubemap
	},

	-- optional --
	"Texture" : ["Filename1", "Filename2", ....],		-- the fullpath of your texture file

	-- built-in material --
	-- titanium, chromium, iron, nickel, platinum, copper, palladium, zinc, gold, aluminum, silver --
	-- glass, green_glass, red, green, orange, purple, blue, wall_blue, wall_red, wall_green, wall_white, marble, something_blue, something_red --
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
			"ReducedScatteringCoef" : "0.0 0.0 0.0",	-- each component >= 0.0 --
			"DiffuseTextureId" : "0"					-- 0, 1, ....., sizeof(Texture) - 1 [optional]
			"SpecularTextureId" : "0"					-- 0, 1, ....., sizeof(Texture) - 1 [optional]
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
			"Material" : ["XXXX", "XXXX", ...],			-- The name of material for each group in obj file. when the material you specified is less than the number of group, the extra group will be set as the last material. on the contrary, the extra material will be neglected(user declared or built-in material)
			"Path" : "XXXX\\YYYY",						-- fullpath of your obj file
			"Position" : "0.0 0.0 0.0",
			"Scale" : "1.0 1.0 1.0",					-- each component > 0.0 --
			"Rotate" : "0.0 0.0 0.0"
		},
		...
	]
}
*/

class scene_parser
{
private:
	nlohmann::json m_json_parser;

	std::vector<std::string> m_scene_files;

	cube_map_loader m_cube_map_loader;

	triangle_mesh m_triangle_mesh;

	texture_wrapper* m_mesh_textures = nullptr;
	int m_textures_num = 0;
	texture_wrapper* m_mesh_textures_device = nullptr;

	sphere* m_spheres = nullptr;
	int m_sphere_num = 0;
	sphere* m_spheres_device = nullptr;

	bool m_is_loaded = false;

public:
	~scene_parser();

	std::vector<std::string> set_scene_file_directory(const std::string& scene_file_directory);

	bool load_scene(int index);
	void unload_scene();

	bool create_scene_data_device(bvh_build_method bvh_build_method);
	void release_scene_data_device();

	cube_map* get_cube_map_device_ptr();
	triangle* get_triangles_device_ptr();
	bvh_node_device** get_bvh_node_device_ptr();
	sphere* get_sphere_device_ptr();
	texture_wrapper* get_mesh_texture_device_ptr();

	int get_mesh_num() const;
	int get_triangles_num() const;
	int get_mesh_triangle_num(int index) const;
	int get_mesh_vertices_num(int index) const;
	int get_sphere_num() const;
	float3 get_mesh_position(int index) const;
	std::vector<material> get_mesh_material(int index) const;
	float3 get_mesh_scale(int index) const;
	float3 get_mesh_rotate(int index) const;
	float3 get_mesh_rotate_applied(int index) const;
	int get_mesh_shape_num(int index) const;
	sphere get_sphere(int index) const;

	void set_sphere_device(int index, const sphere& sphere);
	void set_mesh_material_device(int index, std::vector<material>& mats);
	void set_mesh_transform_device(
		int index,
		const float3& position,
		const float3& scale,
		std::function<void(const glm::mat4&, glm::mat4&, bvh_node_device*, bvh_node_device*)> bvh_update_function
	);
	void set_mesh_rotate(int index, const float3& rotate);
	void apply_mesh_rotate(int index);

private:
	void init_default_material(std::map<std::string, material>& materials);

	float3 parse_float3(const std::string& text);
	float parse_float(const std::string& text);
	bool parse_bool(const std::string& text);
	int parse_int(const std::string& text);
};

#endif // !__SCENE_PARSER__