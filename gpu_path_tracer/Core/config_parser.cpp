#include "Core\config_parser.h"

config_parser::~config_parser()
{
	unload_config();
}

bool config_parser::load_config(const std::string& filename)
{
	if (m_is_loaded)
	{
		return false;
	}
	try
	{
		std::ifstream config_file(filename, std::ios::in);
		config_file >> m_json_parser;
	}
	catch (nlohmann::detail::parse_error error)
	{
		std::cout << "[Error]" << error.what() << std::endl;
		return false;
	}

	auto height = m_json_parser[TOKEN_CONFIG_HEIGHT];
	auto width = m_json_parser[TOKEN_CONFIG_WIDTH];
	auto use_fullscreen = m_json_parser[TOKEN_CONFIG_FULLSCREEN];
	auto block_size = m_json_parser[TOKEN_CONFIG_BLOCK_SIZE];
	auto max_block_size = m_json_parser[TOKEN_CONFIG_MAX_BLOCK_SIZE];
	auto max_tracer_depth = m_json_parser[TOKEN_CONFIG_MAX_TRACER_DEPTH];
	auto vector_bias_length = m_json_parser[TOKEN_CONFIG_VECTOR_BIAS_LENGTH];
	auto energy_exist_threshold = m_json_parser[TOKEN_CONFIG_ENERGY_EXIST_THRESHOLD];
	auto sss_threshold = m_json_parser[TOKEN_CONFIG_SSS_THRESHOLD];
	auto use_sky_box = m_json_parser[TOKEN_CONFIG_SKY_BOX];
	auto use_bilinear = m_json_parser[TOKEN_CONFIG_SKY_BOX_BILINEAR_SAMPLE];
	auto use_sky = m_json_parser[TOKEN_CONFIG_SKY];
	auto gamma_correction = m_json_parser[TOKEN_CONFIG_GAMMA_CORRECTION];
	auto use_anti_alias = m_json_parser[TOKEN_CONFIG_ANTI_ALIAS];
	auto fov = m_json_parser[TOKEN_CONFIG_FOV];
	auto bvh_leaf_node_triangle_num = m_json_parser[TOKEN_CONFIG_BVH_LEAF_NODE_TRIANGLE_NUM];
	auto bvh_bucket_max_divide_internal_num = m_json_parser[TOKEN_CONFIG_BVH_BUCKET_MAX_DIVIDE_INTERNAL_NUM];
	auto bvh_build_block_size = m_json_parser[TOKEN_CONFIG_BVH_BUILD_BLOCK_SIZE];

	CHECK_PROPERTY(Config, height, TOKEN_CONFIG_HEIGHT);
	CHECK_PROPERTY(Config, width, TOKEN_CONFIG_WIDTH);
	CHECK_PROPERTY(Config, use_fullscreen, TOKEN_CONFIG_FULLSCREEN);
	CHECK_PROPERTY(Config, block_size, TOKEN_CONFIG_BLOCK_SIZE);
	CHECK_PROPERTY(Config, max_block_size, TOKEN_CONFIG_MAX_BLOCK_SIZE);
	CHECK_PROPERTY(Config, max_tracer_depth, TOKEN_CONFIG_MAX_TRACER_DEPTH);
	CHECK_PROPERTY(Config, vector_bias_length, TOKEN_CONFIG_VECTOR_BIAS_LENGTH);
	CHECK_PROPERTY(Config, energy_exist_threshold, TOKEN_CONFIG_ENERGY_EXIST_THRESHOLD);
	CHECK_PROPERTY(Config, sss_threshold, TOKEN_CONFIG_SSS_THRESHOLD);
	CHECK_PROPERTY(Config, use_sky_box, TOKEN_CONFIG_SKY_BOX);
	CHECK_PROPERTY(Config, use_bilinear, TOKEN_CONFIG_SKY_BOX_BILINEAR_SAMPLE);
	CHECK_PROPERTY(Config, use_sky, TOKEN_CONFIG_SKY);
	CHECK_PROPERTY(Config, gamma_correction, TOKEN_CONFIG_GAMMA_CORRECTION);
	CHECK_PROPERTY(Config, use_anti_alias, TOKEN_CONFIG_ANTI_ALIAS);
	CHECK_PROPERTY(Config, fov, TOKEN_CONFIG_FOV);
	CHECK_PROPERTY(Config, bvh_leaf_node_triangle_num, TOKEN_CONFIG_BVH_LEAF_NODE_TRIANGLE_NUM);
	CHECK_PROPERTY(Config, bvh_bucket_max_divide_internal_num, TOKEN_CONFIG_BVH_BUCKET_MAX_DIVIDE_INTERNAL_NUM);
	CHECK_PROPERTY(Config, bvh_build_block_size, TOKEN_CONFIG_BVH_BUILD_BLOCK_SIZE);

	std::string height_str = height;
	std::string width_str = width;
	std::string use_fullscreen_str = use_fullscreen;
	std::string block_size_str = block_size;
	std::string max_block_size_str = max_block_size;
	std::string max_tracer_depth_str = max_tracer_depth;
	std::string vector_bias_length_str = vector_bias_length;
	std::string energy_exist_threshold_str = energy_exist_threshold;
	std::string sss_threshold_str = sss_threshold;
	std::string use_sky_box_str = use_sky_box;
	std::string use_bilinear_str = use_bilinear;
	std::string use_sky_str = use_sky;
	std::string gamma_correction_str = gamma_correction;
	std::string use_anti_alias_str = use_anti_alias;
	std::string fov_str = fov;
	std::string bvh_leaf_node_triangle_num_str = bvh_leaf_node_triangle_num;
	std::string bvh_bucket_max_divide_internal_num_str = bvh_bucket_max_divide_internal_num;
	std::string bvh_build_block_size_str = bvh_build_block_size;

	m_height = parse_int(height_str);
	m_width = parse_int(width_str);
	m_use_fullscreen = parse_bool(use_fullscreen_str);
	m_block_size = parse_int(block_size_str);
	m_max_block_size = parse_int(max_block_size_str);
	m_max_tracer_depth = parse_int(max_tracer_depth_str);
	m_vector_bias_length = parse_float(vector_bias_length_str);
	m_energy_exist_threshold = parse_float(energy_exist_threshold_str);
	m_sss_threshold = parse_float(sss_threshold_str);
	m_use_sky_box = parse_bool(use_sky_box_str);
	m_use_bilinear = parse_bool(use_bilinear_str);
	m_use_sky = parse_bool(use_sky_str);
	m_gamma_correction = parse_bool(gamma_correction_str);
	m_use_anti_alias = parse_bool(use_anti_alias_str);
	m_fov = parse_bool(fov_str);
	m_bvh_leaf_node_triangle_num = parse_int(bvh_leaf_node_triangle_num_str);
	m_bvh_bucket_max_divide_internal_num = parse_int(bvh_bucket_max_divide_internal_num_str);
	m_bvh_build_block_size = parse_int(bvh_build_block_size_str);

	m_is_loaded = true;

	return true;
}

void config_parser::unload_config()
{
	m_width = 1024;
	m_height = 768;
	m_use_fullscreen = false;
	m_is_loaded = false;
	m_block_size = 1024;
	m_max_block_size = 1024;
	m_max_tracer_depth = 20;
	m_vector_bias_length = 0.0001f;
	m_energy_exist_threshold = 0.000001f;
	m_sss_threshold = 0.000001f;
	m_use_sky_box = true;
	m_use_bilinear = true;
	m_use_sky = false;
	m_gamma_correction = true;
	m_use_anti_alias = true;
	m_fov = 45.0f;
	m_bvh_leaf_node_triangle_num = 1;
	m_bvh_bucket_max_divide_internal_num = 12;
	m_bvh_build_block_size = 32;
}

configuration* config_parser::get_config_device_ptr()
{
	return m_config_device;
}

void config_parser::create_config_device_data()
{
	CUDA_CALL(cudaMallocManaged((void**)&m_config_device, sizeof(configuration)));
	printf("DRIVER MIGHT BE CRASHED HERE [%s, %d]...", __FILE__, __LINE__);
	m_config_device->width = m_width;
	m_config_device->use_fullscreen = m_use_fullscreen;
	m_config_device->height = m_height;
	m_config_device->block_size = m_block_size;
	m_config_device->max_block_size = m_max_block_size;
	m_config_device->max_tracer_depth = m_max_tracer_depth;
	m_config_device->vector_bias_length = m_vector_bias_length;
	m_config_device->energy_exist_threshold = m_energy_exist_threshold;
	m_config_device->sss_threshold = m_sss_threshold;
	m_config_device->use_sky_box = m_use_sky_box;
	m_config_device->use_bilinear = m_use_bilinear;
	m_config_device->use_sky = m_use_sky;
	m_config_device->gamma_correction = m_gamma_correction;
	m_config_device->use_anti_alias = m_use_anti_alias;
	m_config_device->fov = m_fov;
	m_config_device->bvh_leaf_node_triangle_num = m_bvh_leaf_node_triangle_num;
	m_config_device->bvh_bucket_max_divide_internal_num = m_bvh_bucket_max_divide_internal_num;
	m_config_device->bvh_build_block_size = m_bvh_build_block_size;
	printf("IT'S OK THIS TIME.\n");
}

void config_parser::release_config_device_data()
{
	if (m_config_device != nullptr)
	{
		CUDA_CALL(cudaFree(m_config_device));
		m_config_device = nullptr;
	}
}

float config_parser::parse_float(const std::string& text)
{
	std::istringstream stream(text);
	float value;
	stream >> value;
	return value;
}

int config_parser::parse_int(const std::string& text)
{
	std::istringstream stream(text);
	int value;
	stream >> value;
	return value;
}

bool config_parser::parse_bool(const std::string& text)
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
