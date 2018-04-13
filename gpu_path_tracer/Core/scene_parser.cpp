#include "Core\scene_parser.h"

scene_parser::~scene_parser()
{
	release_scene_data_device();
	unload_scene();
}

std::vector<std::string> scene_parser::set_scene_file_directory(const std::string& scene_file_directory)
{
	m_scene_files.clear();

	std::string suffix = "*.json";
	struct _finddata_t file_info;

	auto file = _findfirst((scene_file_directory + "\\" + suffix).c_str(), &file_info);

	do
	{
		if (strcmp(file_info.name, ".") != 0 && strcmp(file_info.name, "..") != 0)
		{
			m_scene_files.push_back(scene_file_directory + "\\" + file_info.name);
		}

	} while (_findnext(file, &file_info) == 0);
	_findclose(file);

	if (m_scene_files.size() == 0)
	{
		std::cout << "[Warn]There exists no scene file!" << std::endl;
	}

	return m_scene_files;
}

bool scene_parser::load_scene(int index)
{
	if (m_is_loaded || index < 0 || index >= m_scene_files.size())
	{
		return false;
	}

	try
	{
		std::ifstream scene_file(m_scene_files[index], std::ios::in);
		scene_file >> m_json_parser;
	}
	catch (nlohmann::detail::parse_error error)
	{
		std::cout << "[Error]" << error.what() << std::endl;
		return false;
	}

	std::vector<std::string> cubemap_pathes(6);
	std::vector<std::string> texture_pathes;

	std::map<std::string, material> materials;

	std::vector<sphere> spheres;
	std::vector<std::string> spheres_mat;

	std::vector<std::string> meshes_path;
	std::vector<std::vector<std::string>> meshes_mat;
	std::vector<float3> meshes_position;
	std::vector<float3> meshes_scale;
	std::vector<float3> meshes_rotate;

	init_default_material(materials);

	//Step.1 Load data from file

	auto sphere_object = m_json_parser[TOKEN_OBJECT_SPHERE];
	auto mesh_object = m_json_parser[TOKEN_OBJECT_MESH];
	auto background_object = m_json_parser[TOKEN_BACKGROUND];
	auto material_object = m_json_parser[TOKEN_MATERIAL];
	auto texture_object = m_json_parser[TOKEN_TEXTURE];

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

	//Texture
	if (!texture_object.is_null())
	{
		if (!texture_object.is_array())
		{
			std::cout << "[Error]Texture must be array!" << std::endl;
			return false;
		}

		for (auto texture_element : texture_object)
		{
			std::string texture_path = texture_element;
			texture_pathes.push_back(texture_path);
		}
	}

	//Material
	if (!material_object.is_null())
	{
		if (!material_object.is_array())
		{
			std::cout << "[Error]Material must be array!" << std::endl;
			return false;
		}

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
			auto diffuse_texture_id = material_element[TOKEN_MATERIAL_DIFFUSE_TEXTURE_ID];

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
				},
				-1
			};

			if (!diffuse_texture_id.is_null())
			{
				std::string diffuse_texture_id_str = diffuse_texture_id;
				int diffuse_texture_id = parse_int(diffuse_texture_id_str);

				if (diffuse_texture_id != -1 && (diffuse_texture_id >= texture_pathes.size() || diffuse_texture_id < 0))
				{
					std::cout << "[Error]Materail <" << name_str << ">: Texture index out of range!" << std::endl;
					return false;
				}

				mat.diffuse_texture_id = diffuse_texture_id;
			}

			if (mat.is_transparent && mat.medium.extinction_coefficient > 0.0f)
			{
				std::cout << "[Error]Materail <" << name_str << ">: Extinction coefficient of transparent material should be zero!" << std::endl;
				return false;
			}

			materials[name_str] = mat;
		}
	}

	//Sphere
	if (!sphere_object.is_null())
	{
		if (!sphere_object.is_array())
		{
			std::cout << "[Error]Sphere must be array!" << std::endl;
			return false;
		}

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
			sphere.mat = material::get_default_material();

			spheres.push_back(sphere);
			spheres_mat.push_back(material_str);
		}
	}

	//Mesh
	if (!mesh_object.is_null())
	{
		if (!mesh_object.is_array())
		{
			std::cout << "[Error]Mesh must be array!" << std::endl;
			return false;
		}

		for (auto mesh_element : mesh_object)
		{
			auto path = mesh_element[TOKEN_OBJECT_MESH_PATH];
			auto material = mesh_element[TOKEN_OBJECT_MESH_MATERIAL];
			auto position = mesh_element[TOKEN_OBJECT_MESH_POSITION];
			auto scale = mesh_element[TOKEN_OBJECT_MESH_SCALE];
			auto rotate = mesh_element[TOKEN_OBJECT_MESH_ROTATE];

			CHECK_PROPERTY(Mesh, path, TOKEN_OBJECT_MESH_PATH);
			CHECK_PROPERTY(Mesh, material, TOKEN_OBJECT_MESH_MATERIAL);
			CHECK_PROPERTY(Mesh, position, TOKEN_OBJECT_MESH_MATERIAL);
			CHECK_PROPERTY(Mesh, scale, TOKEN_OBJECT_MESH_SCALE);
			CHECK_PROPERTY(Mesh, rotate, TOKEN_OBJECT_MESH_ROTATE);

			std::string path_str = path;
			std::string position_str = position;
			std::string scale_str = scale;
			std::string rotate_str = rotate;

			if (material.is_array())
			{
				std::vector<std::string> mats;
				for (auto mat : material)
				{
					std::string mat_str = mat;
					mats.push_back(mat_str);
				}
				meshes_mat.push_back(mats);
			}
			else
			{
				std::cout << "[Error]Material of mesh must be array!" << std::endl;
				return false;
			}

			meshes_path.push_back(path_str);
			meshes_position.push_back(parse_float3(position_str));
			meshes_scale.push_back(clamp(parse_float3(scale_str), make_float3(0.0f, 0.0f, 0.0f), make_float3(INFINITY, INFINITY, INFINITY)));
			meshes_rotate.push_back(parse_float3(rotate_str));
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

	//Texture
	uint width, height;
	std::vector<uchar> buffer;
	m_mesh_textures = new texture_wrapper[texture_pathes.size()];
	textures_num = static_cast<int>(texture_pathes.size());
	memset(m_mesh_textures, 0, texture_pathes.size() * sizeof(texture_wrapper));
	for (auto i = 0; i < texture_pathes.size(); i++)
	{
		if (image_loader::load_image(texture_pathes[i], width, height, buffer))
		{
			m_mesh_textures[i].width = width;
			m_mesh_textures[i].height = height;
			m_mesh_textures[i].pixels = new uchar[width * height * 4];
			memcpy(m_mesh_textures[i].pixels, buffer.data(), width * height * 4 * sizeof(uchar));
		}
		else
		{
			m_cube_map_loader.unload_data();
			for (auto j = 0; j < i; j++)
			{
				SAFE_DELETE(m_mesh_textures[j].pixels);
			}
			SAFE_DELETE_ARRAY(m_mesh_textures);
			m_mesh_textures = nullptr;
			std::cout << "[Error]Texture " << texture_pathes[i] << " load fail." << std::endl;
			return false;
		}
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

	for (auto mesh_mats : meshes_mat)
	{
		for (auto mesh_mat : mesh_mats)
		{
			if (materials.find(mesh_mat) == materials.end())
			{
				std::cout << "[Error]Material <" << mesh_mat << "> not found!" << std::endl;
				error = true;
			}
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
		std::vector<material*> mesh_mats;
		for (auto mesh_mat : meshes_mat[i])
		{
			mesh_mats.push_back(copy_material(materials[mesh_mat]));
		}

		error = !m_triangle_mesh.load_obj(meshes_path[i], meshes_position[i], meshes_scale[i], meshes_rotate[i], mesh_mats);

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

void scene_parser::unload_scene()
{
	printf("[Info]Unload scene data.\n");
	if (m_is_loaded)
	{
		m_is_loaded = false;
		SAFE_DELETE_ARRAY(m_spheres);
		m_spheres = nullptr;
		m_cube_map_loader.unload_data();
		m_triangle_mesh.unload_obj();
		m_sphere_num = 0;
		for (auto j = 0; j < textures_num; j++)
		{
			SAFE_DELETE(m_mesh_textures[j].pixels);
		}
		SAFE_DELETE_ARRAY(m_mesh_textures);
		m_mesh_textures = nullptr;
		textures_num = 0;
	}
}

bool scene_parser::create_scene_data_device()
{
	if (!m_is_loaded)
	{
		return false;
	}

	double time;

	if (!m_cube_map_loader.create_cube_device_data())
	{
		m_cube_map_loader.release_cube_device_data();
		return false;
	}

	if (m_sphere_num > 0)
	{
		printf("[Info]Copy sphere data to gpu...\n");
		TIME_COUNT_CALL_START();
		CUDA_CALL(cudaMallocManaged((void**)&m_spheres_device, m_sphere_num * sizeof(sphere)));
		CUDA_CALL(cudaMemcpy(m_spheres_device, m_spheres, m_sphere_num * sizeof(sphere), cudaMemcpyDefault));
		TIME_COUNT_CALL_END(time);
		printf("[Info]Completed, time consuming: %.4f ms\n", time);
	}

	if (textures_num > 0)
	{
		printf("[Info]Copy texture data to gpu...\n");
		TIME_COUNT_CALL_START();

		CUDA_CALL(cudaMallocManaged((void**)&m_mesh_textures_device, textures_num * sizeof(texture_wrapper)));
		for (auto i = 0; i < textures_num; i++)
		{
			m_mesh_textures_device[i].width = m_mesh_textures[i].width;
			m_mesh_textures_device[i].height = m_mesh_textures[i].height;
			CUDA_CALL(cudaMalloc((void**)&m_mesh_textures_device[i].pixels, m_mesh_textures[i].width * m_mesh_textures[i].height * 4 * sizeof(uchar)));
			CUDA_CALL(cudaMemcpy(m_mesh_textures_device[i].pixels, m_mesh_textures[i].pixels, m_mesh_textures[i].width * m_mesh_textures[i].height * 4 * sizeof(uchar), cudaMemcpyHostToDevice));
		}

		TIME_COUNT_CALL_END(time);
		printf("[Info]Completed, time consuming: %.4f ms\n", time);
	}

	if (m_triangle_mesh.get_mesh_num() > 0)
	{
		if (!m_triangle_mesh.create_mesh_device_data())
		{
			m_cube_map_loader.release_cube_device_data();
			m_triangle_mesh.release_mesh_device_data();
			return false;
		}

		if (m_triangle_mesh.get_triangles_device() != nullptr && m_cube_map_loader.get_cube_map_device() != nullptr)
		{
			return m_triangle_mesh.create_bvh_device_data();
		}
	}

	return true;
}

void scene_parser::release_scene_data_device()
{
	printf("[Info]Release scene device data.\n");
	m_cube_map_loader.release_cube_device_data();
	m_triangle_mesh.release_mesh_device_data();
	m_triangle_mesh.release_bvh_device_data();
	if (m_spheres_device != nullptr)
	{
		CUDA_CALL(cudaFree(m_spheres_device));
		m_spheres_device = nullptr;
	}
}

cube_map* scene_parser::get_cube_map_device_ptr()
{
	return m_cube_map_loader.get_cube_map_device();
}

triangle* scene_parser::get_triangles_device_ptr()
{
	return m_triangle_mesh.get_triangles_device();
}

bvh_node_device** scene_parser::get_bvh_node_device_ptr()
{
	return m_triangle_mesh.get_bvh_node_device();
}

sphere* scene_parser::get_sphere_device_ptr()
{
	return m_spheres_device;
}

texture_wrapper * scene_parser::get_mesh_texture_device_ptr()
{
	return m_mesh_textures_device;
}

int scene_parser::get_mesh_num() const
{
	return m_triangle_mesh.get_mesh_num();
}

int scene_parser::get_triangles_num() const
{
	return m_triangle_mesh.get_total_triangle_num();
}

int scene_parser::get_mesh_triangle_num(int index) const
{
	return m_triangle_mesh.get_triangle_num(index);
}

int scene_parser::get_mesh_vertices_num(int index) const
{
	return m_triangle_mesh.get_vertex_num(index);
}

int scene_parser::get_sphere_num() const
{
	return m_sphere_num;
}

float3 scene_parser::get_mesh_position(int index) const
{
	return m_triangle_mesh.get_position(index);
}

std::vector<material> scene_parser::get_mesh_material(int index) const
{
	return m_triangle_mesh.get_material(index);
}

float3 scene_parser::get_mesh_scale(int index) const
{
	return m_triangle_mesh.get_scale(index);
}

float3 scene_parser::get_mesh_rotate(int index) const
{
	return m_triangle_mesh.get_rotate(index);
}

float3 scene_parser::get_mesh_rotate_applied(int index) const
{
	return m_triangle_mesh.get_rotate_applied(index);
}

int scene_parser::get_mesh_shape_num(int index) const
{
	return m_triangle_mesh.get_shape_num(index);
}

sphere scene_parser::get_sphere(int index) const
{
	return m_spheres_device[index];
}

void scene_parser::set_sphere_device(int index, const sphere& sphere)
{
	m_spheres_device[index] = sphere;
}

void scene_parser::set_mesh_material_device(int index, std::vector<material>& mats)
{
	m_triangle_mesh.set_material_device(index, mats);
}

void scene_parser::set_mesh_transform_device(
	int index,
	const float3& position,
	const float3& scale,
	std::function<void(const glm::mat4&, glm::mat4&, bvh_node_device*, bvh_node_device*)> bvh_update_function
)
{
	m_triangle_mesh.set_transform_device(index, position, scale, bvh_update_function);
}

void scene_parser::set_mesh_rotate(int index, const float3& rotate)
{
	m_triangle_mesh.set_rotate(index, rotate);
}

void scene_parser::apply_mesh_rotate(int index)
{
	m_triangle_mesh.apply_rotate(index);
}

void scene_parser::init_default_material(std::map<std::string, material>& materials)
{
	materials.clear();

	materials.insert(std::make_pair("titanium", material_data::metal::titanium()));
	materials.insert(std::make_pair("chromium", material_data::metal::chromium()));
	materials.insert(std::make_pair("iron", material_data::metal::iron()));
	materials.insert(std::make_pair("nickel", material_data::metal::nickel()));
	materials.insert(std::make_pair("platinum", material_data::metal::platinum()));
	materials.insert(std::make_pair("copper", material_data::metal::copper()));
	materials.insert(std::make_pair("palladium", material_data::metal::palladium()));
	materials.insert(std::make_pair("zinc", material_data::metal::zinc()));
	materials.insert(std::make_pair("gold", material_data::metal::gold()));
	materials.insert(std::make_pair("aluminum", material_data::metal::aluminum()));
	materials.insert(std::make_pair("silver", material_data::metal::silver()));

	materials.insert(std::make_pair("glass", material_data::dielectric::glass()));
	materials.insert(std::make_pair("green_glass", material_data::dielectric::green_glass()));
	materials.insert(std::make_pair("diamond", material_data::dielectric::diamond()));
	materials.insert(std::make_pair("red", material_data::dielectric::red()));
	materials.insert(std::make_pair("green", material_data::dielectric::green()));
	materials.insert(std::make_pair("orange", material_data::dielectric::orange()));
	materials.insert(std::make_pair("purple", material_data::dielectric::purple()));
	materials.insert(std::make_pair("wall_blue", material_data::dielectric::wall_blue()));
	materials.insert(std::make_pair("blue", material_data::dielectric::blue()));
	materials.insert(std::make_pair("marble", material_data::dielectric::marble()));
	materials.insert(std::make_pair("something_blue", material_data::dielectric::something_blue()));
	materials.insert(std::make_pair("something_red", material_data::dielectric::something_red()));

	materials.insert(std::make_pair("light", material_data::dielectric::light()));
}

float3 scene_parser::parse_float3(const std::string& text)
{
	std::istringstream stream(text);
	float x, y, z;
	stream >> x >> y >> z;
	return make_float3(x, y, z);
}

float scene_parser::parse_float(const std::string& text)
{
	std::istringstream stream(text);
	float value;
	stream >> value;
	return value;
}

bool scene_parser::parse_bool(const std::string& text)
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

int scene_parser::parse_int(const std::string& text)
{
	std::istringstream stream(text);
	int value;
	stream >> value;
	return value;
}
