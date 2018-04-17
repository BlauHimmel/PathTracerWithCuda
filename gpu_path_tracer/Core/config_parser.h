#pragma once

#ifndef __CONFIG_PARSER__
#define __CONFIG_PARSER__

#include "lib\json\json.hpp"

#include "Others\utilities.hpp"
#include "Core\configuration.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#define TOKEN_CONFIG_WIDTH "Width"
#define TOKEN_CONFIG_HEIGHT "Height"
#define TOKEN_CONFIG_FULLSCREEN "FullScreen"
#define TOKEN_CONFIG_BLOCK_SIZE "BlockSize"
#define TOKEN_CONFIG_MAX_BLOCK_SIZE "MaxBlockSize"
#define TOKEN_CONFIG_MAX_TRACER_DEPTH "MaxDepth"
#define TOKEN_CONFIG_VECTOR_BIAS_LENGTH "BiasLength"
#define TOKEN_CONFIG_ENERGY_EXIST_THRESHOLD "EnergyThreshold"
#define TOKEN_CONFIG_SSS_THRESHOLD "SSSThreshold"
#define TOKEN_CONFIG_SKY_BOX "Skybox"
#define TOKEN_CONFIG_SKY_BOX_BILINEAR_SAMPLE "BilinearSample"
#define TOKEN_CONFIG_SKY "Sky"
#define TOKEN_CONFIG_GAMMA_CORRECTION "GammaCorrection"
#define TOKEN_CONFIG_ANTI_ALIAS "AntiAlias"
#define TOKEN_CONFIG_FOV "FOV"
#define TOKEN_CONFIG_BVH_LEAF_NODE_TRIANGLE_NUM "BvhLeafNodeTriangleNum"
#define TOKEN_CONFIG_BVH_BUCKET_MAX_DIVIDE_INTERNAL_NUM "BvhBucketMaxDivideInternalNum"
#define TOKEN_CONFIG_BVH_BUILD_BLOCK_SIZE "BvhBuildBlockSize"

/*
{
	"Width" : "1024",
	"Height" : "768",
	"FullScreen" : "true",
	"BlockSize" : "64",
	"MaxBlockSize" : "1024",
	"MaxDepth" : "20",
	"BiasLength" : "0.0001",
	"EnergyThreshold" : "0.000001",
	"SSSThreshold" : "0.000001",
	"Skybox" : "true",
	"BilinearSample" : "true",
	"Sky" : "false",
	"GammaCorrection" : "true",
	"AntiAlias" : "true",
	"FOV" : "45",
	"BvhLeafNodeTriangleNum" : "1",
	"BvhBucketMaxDivideInternalNum" : "12",
	"BvhBuildBlockSize" : "32
}
*/

class config_parser
{
private:
	//============================================
	int m_width = 1024;
	int m_height = 768;
	bool m_use_fullscreen = false;
	int m_block_size = 64;
	int m_max_block_size = 1024;
	int m_max_tracer_depth = 20;
	float m_vector_bias_length = 0.0001f;
	float m_energy_exist_threshold = 0.000001f;
	float m_sss_threshold = 0.000001f;
	bool m_use_sky_box = true;
	bool m_use_bilinear = false;
	bool m_use_sky = false;
	bool m_gamma_correction = true;
	bool m_use_anti_alias = true;
	float m_fov = 45.0f;
	int m_bvh_leaf_node_triangle_num = 1;
	int m_bvh_bucket_max_divide_internal_num = 12;
	int m_bvh_build_block_size = 32;
	//============================================

	configuration* m_config_device = nullptr;

	bool m_is_loaded = false;

	nlohmann::json m_json_parser;

public:
	~config_parser();

	bool load_config(const std::string& filename);
	void unload_config();

	configuration* get_config_device_ptr();

	void create_config_device_data();
	void release_config_device_data();

private:
	float parse_float(const std::string& text);
	int parse_int(const std::string& text);
	bool parse_bool(const std::string& text);
};

#endif // !__CONFIG_PARSER__
