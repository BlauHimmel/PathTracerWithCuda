#pragma once

#ifndef __CONFIG_PARSER__
#define __CONFIG_PARSER__

#include "lib\json\json.hpp"

#include "utilities.hpp"
#include "configuration.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#define TOKEN_CONFIG_WIDTH "Width"
#define TOKEN_CONFIG_HEIGHT "Height"
#define TOKEN_CONFIG_FULLSCREEN "FullScreen"
#define TOKEN_CONFIG_BLOCK_SIZE "BlockSize"
#define TOKEN_CONFIG_MAX_TRACER_DEPTH "MaxDepth"
#define TOKEN_CONFIG_VECTOR_BIAS_LENGTH "BiasLength"
#define TOKEN_CONFIG_ENERGY_EXIST_THRESHOLD "EnergyThreshold"
#define TOKEN_CONFIG_SSS_THRESHOLD "SSSThreshold"
#define TOKEN_CONFIG_SKY_BOX "Skybox"
#define TOKEN_CONFIG_SKY_BOX_BILINEAR_SAMPLE "BilinearSample"
#define TOKEN_CONFIG_GROUND "Ground"

/*
{
	"Width" : "1024",
	"Height" : "768",
	"FullScreen" : "true",
	"BlockSize" : "1024",
	"MaxDepth" : "20",
	"BiasLength" : "0.0001",
	"EnergyThreshold" : "0.000001",
	"SSSThreshold" : "0.000001",
	"Skybox" : "true",
	"BilinearSample" : "true",
	"Ground" : "false"
}
*/

class config_parser
{
private:
	int m_width = 1024;
	int m_height = 768;
	bool m_use_fullscreen = false;
	int m_block_size = 1024;
	int m_max_tracer_depth = 20;
	float m_vector_bias_length = 0.0001f;
	float m_energy_exist_threshold = 0.000001f;
	float m_sss_threshold = 0.000001f;
	bool m_use_sky_box = true;
	bool m_use_ground = true;
	bool m_use_bilinear = false;

	configuration* m_config = nullptr;

	bool m_is_loaded = false;

	nlohmann::json m_json_parser;

public:
	~config_parser();

	bool load_config(const std::string& filename);
	void unload_config();

	configuration* get_config_ptr();

private:
	float parse_float(const std::string& text);
	int parse_int(const std::string& text);
	bool parse_bool(const std::string& text);
};

inline config_parser::~config_parser()
{
	unload_config();
	SAFE_DELETE(m_config);
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
	auto max_tracer_depth = m_json_parser[TOKEN_CONFIG_MAX_TRACER_DEPTH];
	auto vector_bias_length = m_json_parser[TOKEN_CONFIG_VECTOR_BIAS_LENGTH];
	auto energy_exist_threshold = m_json_parser[TOKEN_CONFIG_ENERGY_EXIST_THRESHOLD];
	auto sss_threshold = m_json_parser[TOKEN_CONFIG_SSS_THRESHOLD];
	auto use_sky_box = m_json_parser[TOKEN_CONFIG_SKY_BOX];
	auto use_bilinear = m_json_parser[TOKEN_CONFIG_SKY_BOX_BILINEAR_SAMPLE];
	auto use_ground = m_json_parser[TOKEN_CONFIG_GROUND];

	CHECK_PROPERTY(Config, height, TOKEN_CONFIG_HEIGHT);
	CHECK_PROPERTY(Config, width, TOKEN_CONFIG_WIDTH);
	CHECK_PROPERTY(Config, use_fullscreen, TOKEN_CONFIG_FULLSCREEN);
	CHECK_PROPERTY(Config, block_size, TOKEN_CONFIG_BLOCK_SIZE);
	CHECK_PROPERTY(Config, max_tracer_depth, TOKEN_CONFIG_MAX_TRACER_DEPTH);
	CHECK_PROPERTY(Config, vector_bias_length, TOKEN_CONFIG_VECTOR_BIAS_LENGTH);
	CHECK_PROPERTY(Config, energy_exist_threshold, TOKEN_CONFIG_ENERGY_EXIST_THRESHOLD);
	CHECK_PROPERTY(Config, sss_threshold, TOKEN_CONFIG_SSS_THRESHOLD);
	CHECK_PROPERTY(Config, use_sky_box, TOKEN_CONFIG_SKY_BOX);
	CHECK_PROPERTY(Config, use_bilinear, TOKEN_CONFIG_SKY_BOX_BILINEAR_SAMPLE);
	CHECK_PROPERTY(Config, use_ground, TOKEN_CONFIG_GROUND);

	std::string height_str = height;
	std::string width_str = width;
	std::string use_fullscreen_str = use_fullscreen;
	std::string block_size_str = block_size;
	std::string max_tracer_depth_str = max_tracer_depth;
	std::string vector_bias_length_str = vector_bias_length;
	std::string energy_exist_threshold_str = energy_exist_threshold;
	std::string sss_threshold_str = sss_threshold;
	std::string use_sky_box_str = use_sky_box;
	std::string use_bilinear_str = use_bilinear;
	std::string use_ground_str = use_ground;

	m_height = parse_int(height_str);
	m_width = parse_int(width_str);
	m_use_fullscreen = parse_bool(use_fullscreen_str);
	m_block_size = parse_int(block_size_str);
	m_max_tracer_depth = parse_int(max_tracer_depth_str);
	m_vector_bias_length = parse_float(vector_bias_length_str);
	m_energy_exist_threshold = parse_float(energy_exist_threshold_str);
	m_sss_threshold = parse_float(sss_threshold_str);
	m_use_sky_box = parse_bool(use_sky_box_str);
	m_use_bilinear = parse_bool(use_bilinear_str);
	m_use_ground = parse_bool(use_ground_str);

	m_is_loaded = true;

	return true;
}

inline void config_parser::unload_config()
{
	m_width = 1024;
	m_height = 768;
	m_use_fullscreen = false;
	m_is_loaded = false;
	m_block_size = 1024;
	m_max_tracer_depth = 20;
	m_vector_bias_length = 0.0001f;
	m_energy_exist_threshold = 0.000001f;
	m_sss_threshold = 0.000001f;
	m_use_sky_box = true;
	m_use_bilinear = true;
	m_use_ground = false;
}

inline configuration* config_parser::get_config_ptr()
{
	if (m_config == nullptr)
	{
		m_config = new configuration();
		m_config->width = m_width;
		m_config->height = m_height;
		m_config->use_fullscreen = m_use_fullscreen;
		m_config->block_size = m_block_size;
		m_config->max_tracer_depth = m_max_tracer_depth;
		m_config->vector_bias_length = m_vector_bias_length;
		m_config->energy_exist_threshold = m_energy_exist_threshold;
		m_config->sss_threshold = m_sss_threshold;
		m_config->use_sky_box = m_use_sky_box;
		m_config->use_bilinear = m_use_bilinear;
		m_config->use_ground = m_use_ground;
	}

	return m_config;
}

inline float config_parser::parse_float(const std::string& text)
{
	std::istringstream stream(text);
	float value;
	stream >> value;
	return value;
}

inline int config_parser::parse_int(const std::string& text)
{
	std::istringstream stream(text);
	int value;
	stream >> value;
	return value;
}

inline bool config_parser::parse_bool(const std::string& text)
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

#endif // !__CONFIG_PARSER__
